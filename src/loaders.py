from typing import Dict, Literal, List, Tuple
from torch import Tensor
from tqdm import tqdm
import os
import regex
import mat73
import torch
import pickle

def load_mat(name: str, split_real_imaginary: bool = True, include_metadata: List[Literal["fiber_fractions"]] = []) -> Tuple[Tensor, Tensor]:
    """
        Will load the qspace array from .mat file and translate it into a 'Torch.Tensor'.
        Make sure, that 'PATH_MATFILES' in '.env' points to correct folder!

        name:
            full name of the .mat file.
            No path, just the name, like simulation_results_02.mat
        split_real_imaginary:
            # TODO
        include_metadata:
            # TODO
        return:
            # TODO need to be adjusted to new outputs
            # All samples in the .mat file as 'Torch.Tensor'.
            # Shape: (33, 33, 33, 2)
            # Note: Last dimension is for real (pos0) and imaginary (pos1) part of the number
    """ 
    # Want to check if the name of the file has the desired form
    reg_detector = regex.compile(f"simulation_results_\d\d.mat")
    name_list = reg_detector.findall(name)
    # If yes, we can go ahead
    if name_list == []:
        raise Exception("Used wrong file name. Need to look like: simulation_results_03.mat")   
    
    path = os.environ.get("PATH_MATFILES").replace("/", os.sep) + os.sep + name_list[0]
    number = name_list[0][19:-4]
    d = "results_" + str(number)

    mat_dict = mat73.loadmat(path)

    return_dict = {}
    c = 0
    for s in tqdm(range(0, len(mat_dict[d])), desc="Loading from .mat files"):

        if include_metadata != []:
            sample = {}
            for meta_data in include_metadata:
                sample["tensor"] = torch.from_numpy(mat_dict[d][s]["qspace"])
                sample[meta_data] = torch.from_numpy(mat_dict[d][s][meta_data])

            if split_real_imaginary:
                shape = list(sample["tensor"].shape)
                shape.append(2)

                new = torch.zeros(shape, dtype=torch.float64)

                new[:, :, :, 0] = sample["tensor"].real
                new[:, :, :, 1] = sample["tensor"].imag
                sample["tensor"] = new
        else:
            sample = torch.from_numpy(mat_dict[d][s]["qspace"])
            if split_real_imaginary:
                shape = list(sample.shape)
                shape.append(2)

                new = torch.zeros(shape, dtype=torch.float64)

                new[:, :, :, 0] = sample.real
                new[:, :, :, 1] = sample.imag
                sample = new
            

        return_dict[c] = sample
        c += 1

    return_meta = torch.Tensor([])
    if include_metadata == []:
        return_tensor = torch.cat([return_dict[i].unsqueeze(dim=0) for i in return_dict], dim=0)
    else:
        return_tensor = torch.cat([return_dict[i]["tensor"].unsqueeze(dim=0) for i in return_dict], dim=0)
        for m in include_metadata:
            return_meta = torch.cat([return_dict[i][m].unsqueeze(dim=0) for i in return_dict], dim=0) 

    return return_tensor, return_meta

def save_pointclouds(dictionary: Dict[str, Dict[int, Tensor]], name: str) -> None:
    """
        TODO
    """
    path_prefix = os.environ.get("PATH_CLOUD").replace("/", os.sep)
    ensure_folder_exists(path_prefix)

    file_path = path_prefix + os.sep + name
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(dictionary, file)
        print(f"Dictionary saved to '{file_path}' successfully.")
    except Exception as e:
        print(f"Error: {e}")

def ensure_folder_exists(folder_path,):
    try:
        # Check if the provided path already exists as a directory
        if not os.path.exists(folder_path):
            # Create the directory if it doesn't exist
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        else:
            print(f"Folder '{folder_path}' already exists.")

    except Exception as e:
        print(f"Error: {e}")

def save_tensor(batch: Tensor, name: str, to: Literal["tensor", "target"]) -> None:
    """
        batch:

    """
    path_prefix = ""
    if to == "tensor":
        path_prefix = os.environ.get("PATH_TENSOR").replace("/", os.sep)
    elif to == "target":
        path_prefix = os.environ.get("PATH_TARGETS").replace("/", os.sep)

    ensure_folder_exists(path_prefix)
    
    torch.save(
        obj=batch,
        f= path_prefix + os.sep + name,
    )

def load_tensor(name: str, of: Literal["tensor", "target"]) -> Tensor:
    """
        # TODO
    
        name:
        
        return:

    """
    path_prefix = ""
    if of == "tensor":
        path_prefix = os.environ.get("PATH_TENSOR").replace("/", os.sep)
    elif of == "target":
        path_prefix = os.environ.get("PATH_TARGETS").replace("/", os.sep)

    return torch.load(f=path_prefix + os.sep + name)

def list_files_in_directory(directory_path) -> List[str]:
    try:
        # Check if the provided path is a directory
        if not os.path.isdir(directory_path):
            raise ValueError("The provided path is not a directory or the directory does not exist yet. (" + directory_path + ")")

        # Get a list of all files and directories within the specified directory
        files = os.listdir(directory_path)

        # Filter out only the files (excluding directories)
        file_names = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]

        return file_names

    except Exception as e:
        print(f"Error: {e}")
        return []

def get_file_names(folder: Literal["mat", "tensor", "target", "cloud"] = "tensor") -> List[str]:
    """
        # TODO
    
        name:
        
        return:
    """

    if folder not in ["mat", "tensor", "target", "cloud"]:
        raise Exception("Only supports on of the following: ['mat', 'tensor', 'target', 'cloud']")

    path = ""
    if folder == "mat":
        # path_prefix = os.environ.get("PATH_MATFILES").replace("/", os.sep)
        return ['simulation_results_01.mat', 'simulation_results_02.mat', 'simulation_results_03.mat',
                'simulation_results_04.mat', 'simulation_results_05.mat', 'simulation_results_06.mat',
                'simulation_results_07.mat', 'simulation_results_08.mat', 'simulation_results_09.mat',
                'simulation_results_10.mat', 'simulation_results_11.mat', 'simulation_results_12.mat',
                'simulation_results_13.mat', 'simulation_results_14.mat', 'simulation_results_15.mat',
                'simulation_results_16.mat', 'simulation_results_17.mat', 'simulation_results_18.mat',
                'simulation_results_19.mat', 'simulation_results_20.mat']
    if folder == "tensor":
        path = os.environ.get("PATH_TENSOR").replace("/", os.sep)
    elif folder == "target":
        path = os.environ.get("PATH_TARGETS").replace("/", os.sep)
    elif folder == "cloud":
        path = os.environ.get("PATH_CLOUD").replace("/", os.sep)
    return list_files_in_directory(path)

def load_batch_for_training(name: str, noise: float = 0) -> Tensor:
    """
        # TODO
    
        name:

        noise:
            # TODO Work in Progess
        
        return:

    """
    tensor, _ = load_mat(name=name, split_real_imaginary=True)
    
    if noise > 0:
        tensor=tensor # TODO
    return tensor

def load_targets_for_training(name: str) -> Tensor:
    """
        # TODO
    
        name:
        
        return:

    """
    return load_tensor(name=name, of="target")