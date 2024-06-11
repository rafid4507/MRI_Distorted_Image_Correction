"""File which let's you run the needed preprocessing tasks"""

from typing import Dict, Literal, Tuple
from torch import Tensor
from tqdm import tqdm
import torch

from src.transforms import calculate_threshold, get_mags
from src.loaders import load_mat, get_file_names, save_pointclouds, save_tensor

def create_pointclouds_batch(name: str, approach: Literal["mean", "0.6", "4xmean"] = "mean") -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    """
        Will load the qspace array from .mat file and translate it into a 'Torch.Tensor'.
        Created Tensor is a point cloud representation of a thresholded voxel sample,
        which we need for Hugh Transfrom and PCA

        name:
            full name of the .mat file.
            No path, just the name, like simulation_results_02.mat
        return:
            # # TODO
            # All samples in the .mat file as 'Torch.Tensor'.
            # Shape: ("coordinates": [x, y, z,] , "fiber_fractions": [])
            # Note:   the coordinates are the indices of the points in the array
            #         fiber_fractions are the same as in the .mat file
    """
    
    samples, labels = load_mat(name=name, split_real_imaginary=True, include_metadata=["fiber_fractions"])
    mag_samples = get_mags(batch=samples)

    num_samples = samples.shape[0]
    results_sample = {}
    results_labels = {}
    for index in tqdm(range(0, num_samples), desc="Creation of Point Clouds"):
        threshold = calculate_threshold(
            voxel=mag_samples[index],
            appproach=approach,
        )

        # Condition to find indices where values > threshold
        results_sample[index] = (mag_samples[index] > threshold).nonzero().to(dtype = torch.float64)
        results_labels[index] = labels[index]

    return results_sample, results_labels

def create_all_pointclouds_batches(approach: Literal["mean", "0.6", "4xmean"] = "mean")-> None:
    """
        Well create the point clouds and save them to their respective folder
        This will only create and save clean point clouds.
    """
    batches = get_file_names("mat")

    for b in batches:
        re_sample, re_labels = create_pointclouds_batch(name=b, approach=approach)
        dic = {"coordinates": re_sample, "fiber_fractions": re_labels}
        
        # Save to dics
        name = b.replace(".mat", "") + "_cloud.pkl"
        save_pointclouds(dictionary=dic, name=name)

def create_targets(name: str) -> None:
    """
        Creating single tensor (samples AND targets) for one .mat file7
        Saving it to the corresponding folder
    """
    # cloud = None
    labels = torch.zeros([500, 2])# TODO Placeholder
    save_tensor(
        labels,
        name=name.replace(".pkl", "") + "_targets",
        to="target"
        )

def create_all_targets() -> None:
    """
        For each clouds batch, it will create the labels, which will be used for the training.
    """
    batches = get_file_names("cloud")
    for b in tqdm(batches, "Creating Targets"):
        create_targets(name=b)

def create(what: Literal["target", "cloud"]):
    """
        TODO
    """
    if what == "cloud": 
        create_all_pointclouds_batches()
    if what == "target":
        create_all_targets()