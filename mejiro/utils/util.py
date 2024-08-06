import datetime
import os
import pickle as _pickle
import shutil
from collections import ChainMap
from glob import glob
import json

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf
from PIL import Image


def write_execution_time(time, script_name, filepath):
    """
    Write the execution time of a script to a file.

    Parameters
    ----------
    time : float
        The execution time in seconds.
    script_name : str
        The name of the script.
    filepath : str
        The path to the file where the execution time will be written.

    Returns
    -------
    None

    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            contents = json.load(f)
        contents[script_name] = time
        with open(filepath, 'w') as f:
            data = json.dumps(contents, indent=4)
            f.write(data)
    else:
        with open(filepath, 'w') as f:
            data = json.dumps({script_name: time}, indent=4)
            f.write(data)


def rotate_array(array, angle, fillcolor='white'):
    """
    Rotate a 2D numpy array by a given angle.

    Parameters
    ----------
    array : numpy.ndarray 
        The input array to be rotated.
    angle : float
        The angle of rotation in degrees.

    Returns
    -------
    numpy.ndarray
        The rotated array.

    """
    pil_image = Image.fromarray(array)
    rotated_pil_image = pil_image.rotate(angle, fillcolor=fillcolor)
    return np.asarray(rotated_pil_image)


def get_kwargs_cosmo(astropy_cosmo):
    """
    Get `kwargs_cosmo` for pyhalo from an Astropy cosmology object.
    """
    return {
        "H0": astropy_cosmo.H0.value,
        "Ob0": astropy_cosmo.Ob0,
        "Om0": astropy_cosmo.Om0,
    }


def save_skypy_config(skypy_config, path):
    # TODO needs to account for serializing e.g. numpy arrays, probably some similar code to deserialization method below
    with open(path, 'w') as file:
        yaml.dump(skypy_config, file)


def load_skypy_config(path):
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown
    )

    with open(path, "r") as file:
        skypy_config = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)

    return skypy_config


def percent_change(a, b):
    """
    Calculate the percentage change between two values.

    Parameters
    ----------
    a : float
        The first value.
    b : float
        The second value.

    Returns
    -------
    float
        The percentage change between the two values.

    Examples
    --------
    >>> percent_change(10, 20)
    100.0

    >>> percent_change(10, 15)
    50.0
    """
    return np.abs(a - b) / a * 100


def percent_difference(a, b):
    """
    Calculate the percentage difference between two values.

    Parameters
    ----------
    a : float
        The first value.
    b : float
        The second value.

    Returns
    -------
    float
        The percentage difference between the two values.

    Examples
    --------
    >>> percent_difference(10, 20)
    100.0

    >>> percent_difference(10, 15)
    50.0
    """
    return np.abs(a - b) / ((a + b) / 2) * 100


def combine_all_csvs(path, prefix="", filename=None):
    """
    Combine all CSV files in a directory into a single DataFrame.

    Parameters
    -----------
    path : str
        The path to the directory containing the CSV files.
    prefix : str, optional
        The prefix of the CSV files to be combined. Default is an empty string.
    filename : str, optional
        The name of the combined CSV file to save. If not provided, the DataFrame will not be saved.

    Returns
    --------
    df_res : pandas.DataFrame
        The combined DataFrame containing all the CSV data.

    Example:
    --------
    >>> combine_all_csvs('/path/to/csvs', 'combined.csv')
    Wrote combined CSV to combined.csv
    <DataFrame object>
    """
    # list all files in directory
    csv_files = glob(os.path.join(path, f"{prefix}*.csv"))

    # concatenate CSVs
    pd_list = [pd.read_csv(f) for f in csv_files]
    df_res = pd.concat(pd_list, ignore_index=True)  # TODO fix FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.

    # save as combined CSV
    if filename is not None:
        df_res.to_csv(filename)
        print(f"Wrote combined CSV to {filename}")

    # return as DataFrame
    return df_res


def check_negative_values(array):
    """
    Check if there are any negative values in the given array or list of arrays.

    Parameters
    ----------
    array : array-like or list of array-like
        The input array or list of arrays to check for negative values.

    Returns
    -------
    bool
        True if there are negative values, False otherwise.

    Notes
    -----
    This function uses numpy's `any` function to check for negative values.

    Examples
    --------
    >>> check_negative_values([1, 2, -3, 4])
    True

    >>> check_negative_values([[1, 2], [3, -4]])
    True

    >>> check_negative_values([1, 2, 3, 4])
    False
    """
    if isinstance(array, list):
        for a in array:
            if np.any(a < 0):
                return True
    else:
        return np.any(array < 0)


def replace_negatives_with_zeros(array):
    """
    Replace negative values in the input array with zeros.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Array with negative values replaced by zeros.
    """
    return np.where(array < 0, 0, array)


def resize_with_pixels_centered(array, oversample_factor):
    """
    Resize the input array with centered pixels using the specified oversample factor.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to be resized.
    oversample_factor : int
        The factor by which to oversample the array. Must be odd.

    Returns
    -------
    numpy.ndarray
        The resized array with centered pixels.

    Raises
    ------
    Exception
        If the oversample factor is not odd.
    Exception
        If the array is not square.

    """
    if oversample_factor % 2 == 0:
        raise Exception("Oversampling factor must be odd")

    x, y = array.shape
    if x != y:
        raise Exception("Array must be square")

    flattened_array = array.flatten()
    oversample_grid = np.zeros((x * oversample_factor, x * oversample_factor))

    k = 0
    for i, row in enumerate(oversample_grid):
        for j, _ in enumerate(row):
            if not (i % oversample_factor) - ((oversample_factor - 1) / 2) == 0:
                continue
            if (j % oversample_factor) - ((oversample_factor - 1) / 2) == 0:
                oversample_grid[i][j] = flattened_array[k]
                k += 1

    return oversample_grid


def center_crop_image(array, shape):
    """
    Crop the input array to the specified shape by centering the image.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to be cropped.
    shape : tuple
        The desired shape of the cropped array.

    Returns
    -------
    numpy.ndarray
        The cropped array.

    Notes
    -----
    If the input array already has the specified shape, it will be returned as is.

    Examples
    --------
    >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> shape = (2, 2)
    >>> center_crop_image(array, shape)
    array([[5, 6],
           [8, 9]])

    >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> shape = (3, 3)
    >>> center_crop_image(array, shape)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    """
    if array.shape == shape:
        return array

    y_out, x_out = shape
    tuple = array.shape
    y, x = tuple[0], tuple[1]
    x_start = (x // 2) - (x_out // 2)
    y_start = (y // 2) - (y_out // 2)
    return array[y_start: y_start + y_out, x_start: x_start + x_out]


def hydra_to_dict(config):
    """
    Convert a Hydra configuration object to a dictionary.

    Parameters
    ----------
    config : OmegaConf.DictConfig
        The Hydra configuration object.

    Returns
    -------
    dict
        A dictionary representation of the Hydra configuration.

    """
    container = OmegaConf.to_container(config, resolve=True)
    return dict(ChainMap(*container))


def print_execution_time(start, stop, return_string=False):
    """
    Print the execution time between two given timestamps.

    Parameters
    ----------
    start : float
        The start timestamp.
    stop : float
        The stop timestamp.

    Returns
    -------
    None

    Examples
    --------
    >>> start = time.time()
    >>> # Some code to measure execution time
    >>> stop = time.time()
    >>> print_execution_time(start, stop)
    Execution time: 0:00:05

    """
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f"Execution time: {execution_time}")
    if return_string:
        return execution_time


def pickle(path, thing):
    """
    Use the `pickle` module to serialize an object and save it to a file. Note that the file will be overwritten if it already exists.

    Parameters
    ----------
    path : str
        The path to the file where the object will be saved.
    thing : object
        The object to be pickled and saved.

    Returns
    -------
    None

    Examples
    --------
    >>> pickle('/path/to/file.pkl', {'key': 'value'})
    """
    with open(path, "wb") as results_file:
        _pickle.dump(thing, results_file)


def unpickle(path):
    """
    Unpickle the object stored in the given file path and return it.

    Parameters
    ----------
    path : str
        The path to the file containing the pickled object.

    Returns
    -------
    object
        The unpickled object.

    """
    with open(path, "rb") as results_file:
        result = _pickle.load(results_file)
    return result


def unpickle_all(dir_path, prefix="", suffix="", limit=None):
    """
    Load and unpickle all files in a directory.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing the files.
    prefix : str, optional
        The prefix of the files to be loaded. Default is an empty string.
    suffix : str, optional
        The suffix of the files to be loaded. Default is an empty string.
    limit : int, optional
        The maximum number of files to be loaded. Default is None.

    Returns
    -------
    list
        A list of unpickled objects.

    """
    file_list = glob(dir_path + f"/{prefix}*{suffix}")
    sorted_list = sorted(file_list)
    if limit is not None:
        return [unpickle(i) for i in sorted_list[:limit] if os.path.isfile(i)]
    else:
        return [unpickle(i) for i in sorted_list if os.path.isfile(i)]


def create_directory_if_not_exists(path):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        The path of the directory to be created.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clear_directory(path):
    """
    Clear all files and directories within the specified path.

    Parameters
    ----------
    path : str
        The path to the directory to be cleared.

    Returns
    -------
    None

    """
    for i in glob(path + "/*"):
        if os.path.isfile(i):
            os.remove(i)
        else:
            shutil.rmtree(i)


def batch_list(list, n):
    """
    Split a list into batches of size n. This method is used for parallel processing.

    Parameters
    ----------
    list : list
        The input list to be split into batches.
    n : int
        The size of each batch.

    Yields
    ------
    list
        A batch of size n from the input list.

    Examples
    --------
    >>> my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> for batch in batch_list(my_list, 3):
    ...     print(batch)
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]
    """
    for i in range(0, len(list), n):
        yield list[i: i + n]


def scientific_notation_string(input):
    """
    Convert a number to a string representation in scientific notation.

    Parameters
    ----------
    input : float
        The number to be converted.

    Returns
    -------
    str
        The string representation of the number in scientific notation.

    Examples
    --------
    >>> scientific_notation_string(1000000)
    '1.00e+06'

    >>> scientific_notation_string(0.000001)
    '1.00e-06'
    """
    return "{:.2e}".format(input)


def delete_if_exists(path):
    """
    Delete a file if it exists.

    Parameters
    ----------
    path : str
        The path to the file to be deleted.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        os.remove(path)
