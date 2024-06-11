import torch
from typing import Literal

# gets the magnitudes
def get_mags(batch: torch.Tensor) -> torch.Tensor:
    """
    Calculate the normalized magnitudes of a voxel for each sample in the batch.

    Parameters:
        batch (torch.Tensor): The input batch tensor of shape (BATCH_SIZE, 33, 33, 33, 2).

    Returns:
        torch.Tensor: The normalized magnitudes of the voxel (shape: BATCH_SIZE, 33, 33, 33).
    """
    if batch.shape[1:] != (33, 33, 33, 2):
        raise Exception("Input tensor has wrong shape. Expected (BATCH_SIZE, 33, 33, 33, 2).")
    
    mags = torch.norm(batch, dim=4)
    for s in range(0, len(mags)):    
        mags[s] = (mags[s]-torch.min(mags[s])) / (torch.max(mags[s])-torch.min(mags[s])) 
    return mags

def calculate_threshold(voxel: torch.Tensor, approach: Literal["mean", "0.6", "4xmean"] = "mean") -> float:
    """
        Calculates the threshold for the point cloud creation.
        Estimates an optimal threshold in the range of parsed voxel for cut-off.

        voxel:
            Magnitude Tensor of shape 33x33x33
        return:
            Estimated Threshold, in range of the voxel numbers 
    """
    if approach == "mean":
        return float(voxel.mean())
    if approach == "0.6":
        return float(0.6)
    if approach == "4xmean":
        return float(4 *  voxel.mean())
    
def apply_noise(batch: torch.Tensor) -> torch.Tensor:
    """
        TODO
    """
    if batch.shape[1:] != (33, 33, 33, 2):
        raise Exception("Input tensor has wrong shape. Expected (BATCH_SIZE, 33, 33, 33, 2).")
    return batch