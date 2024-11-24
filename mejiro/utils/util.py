import datetime
import json
import os
import pickle as _pickle
import shutil
import warnings
from collections import ChainMap
from glob import glob

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from omegaconf import OmegaConf


def smallest_non_negative_element(array):
    """
    Find the smallest non-negative element in a given array.

    Parameters
    ----------
    array : numpy.ndarray
        Input array containing numerical elements.

    Returns
    -------
    float or None
        The smallest non-negative element in the array. If no non-negative 
        elements are found, returns None.
    """
    non_negative_elements = array[array >= 0]
    if non_negative_elements.size > 0:
        return np.min(non_negative_elements)
    else:
        return None


def replace_negatives(array, replacement=0):
    if np.any(array < 0):
        array[array < 0] = replacement
        warnings.warn(f"Negative values in array have been replaced with {replacement}.")
    return array


def create_centered_box(N, box_size):
    """
    Create an NxN array with a centered box of True values.
    
    Parameters
    ----------
    N : int
        The size of the outer array. Must be an odd number.
    box_size : int
        The size of the centered box. Must be an odd number and less than or equal to N.

    Returns
    -------
    numpy.ndarray
        An NxN array with a centered box of True values and the rest False.

    Raises
    ------
    ValueError
        If N is not an odd number.
        If box_size is not an odd number.
        If box_size is greater than N.
    """
    if N % 2 == 0:
        raise ValueError("N must be an odd number")
    if box_size % 2 == 0:
        raise ValueError("box_size must be an odd number")
    if box_size > N:
        raise ValueError("box_size must be less than or equal to N")
    
    # Create an NxN array of False
    array = np.full((N, N), False, dtype=bool)
    
    # Find the coordinates of the centered inner box
    center = N // 2
    half_box_size = box_size // 2
    
    # Set the inner box to True
    array[center-half_box_size:center+half_box_size+1, center-half_box_size:center+half_box_size+1] = True
    
    return array


def create_centered_circle(N, radius):
    """
    Create an NxN boolean array with a centered circle of True values.

    Parameters
    ----------
    N : int
        The size of the NxN array. Must be an odd number.
    radius : float
        The radius of the circle. Must be a positive number and less than or equal to N//2.
        
    Returns
    -------
    numpy.ndarray
        An NxN boolean array with a centered circle of True values.

    Raises
    ------
    ValueError
        If N is not an odd number.
        If radius is not a positive number.
        If radius is greater than N//2.
    """
    if N % 2 == 0:
        raise ValueError("N must be an odd number")
    if radius <= 0:
        raise ValueError("Radius must be a positive number")
    if radius > N // 2:
        raise ValueError(f"Radius ({radius:.2f})must be less than or equal to N//2 ({N // 2:.2f})")
    
    # Create an NxN array of False
    array = np.full((N, N), False, dtype=bool)
    
    # Find the center of the array
    center = (N // 2, N // 2)
    
    # Set the circular region to True
    for i in range(N):
        for j in range(N):
            # Calculate the distance from the center
            if np.sqrt((i - center[0])**2 + (j - center[1])**2) <= radius:
                array[i, j] = True
    
    return array


def all_arrays_equal(iterator):
    """
    Check if all arrays in an iterator are equal.

    Parameters
    ----------
    iterator : iterable
        An iterable containing arrays to be compared.

    Returns
    -------
    bool
        True if all arrays in the iterator are equal, False otherwise.
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(np.array_equal(first, x) for x in iterator)


def make_grid(side_length, num_points):
    """
    Generate a 2D grid of evenly spaced points within a square.

    Parameters
    ----------
    side_length : float
        The length of the sides of the square.
    num_points : int
        The number of points along each axis.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (num_points*num_points, 2) containing the (x, y) coordinates of the grid points.
    """
    # Define the range and number of points for each axis
    x_min, x_max, x_points = -side_length / 2, side_length / 2, num_points
    y_min, y_max, y_points = -side_length / 2, side_length / 2, num_points

    # Generate evenly spaced points for each axis
    x = np.linspace(x_min, x_max, x_points)
    y = np.linspace(y_min, y_max, y_points)

    # Create a 2D grid of points
    X, Y = np.meshgrid(x, y)

    # Stack the coordinates to get a grid of (x, y) pairs
    return np.column_stack([X.ravel(), Y.ravel()])


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

    Parameters
    ----------
    astropy_cosmo : astropy.cosmology.Cosmology
        An instance of an Astropy cosmology object.

    Returns
    -------
    dict
        A dictionary containing the cosmological parameters:
        
        - H0 : float
            The Hubble constant at z=0 in km/s/Mpc.
        - Ob0 : float
            The density of baryonic matter in units of the critical density at z=0.
        - Om0 : float
            The density of non-relativistic matter in units of the critical density at z=0. 
    """
    return {
        "H0": astropy_cosmo.H0.value,
        "Ob0": astropy_cosmo.Ob0,
        "Om0": astropy_cosmo.Om0,
    }


def save_skypy_config(skypy_config, path):
    """
    Save the SkyPy configuration to a YAML file.

    Parameters
    ----------
    skypy_config : dict
        The SkyPy configuration dictionary to be saved.
    path : str
        The file path where the configuration will be saved.

    Notes
    -----
    This function currently does not handle serialization of complex objects
    such as numpy arrays. Future improvements should include handling such cases.
    """
    # TODO needs to account for serializing e.g. numpy arrays, probably some similar code to deserialization method below
    with open(path, 'w') as file:
        yaml.dump(skypy_config, file)


def load_skypy_config(path):
    """
    Load a SkyPy configuration file, ignoring unknown YAML tags.

    This function reads a YAML file from the specified path and loads its content
    into a Python dictionary. It uses a custom YAML loader that ignores any unknown
    tags in the YAML file.

    Parameters
    ----------
    path : str
        The file path to the SkyPy configuration YAML file.

    Returns
    -------
    dict
        A dictionary containing the loaded SkyPy configuration.

    Notes
    -----
    This function uses a custom YAML loader (`SafeLoaderIgnoreUnknown`) that
    ignores unknown tags in the YAML file. This can be useful when the YAML file
    contains tags that are not recognized by the default YAML loader.
    """
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    SafeLoaderIgnoreUnknown.add_constructor(
        None, SafeLoaderIgnoreUnknown.ignore_unknown
    )

    with open(path, "r") as file:
        skypy_config = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)

    return skypy_config


def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float
        The radial distance from the origin.
    theta : float
        The angle in radians from the positive x-axis.

    Returns
    -------
    x : float
        The x-coordinate in Cartesian coordinates.
    y : float
        The y-coordinate in Cartesian coordinates.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def percent_change(old, new):
    """
    Calculate the percent change between two values.

    Parameters
    ----------
    old : float
        The initial value.
    new : float
        The new value.

    Returns
    -------
    float
        The percent change from the old value to the new value.

    Examples
    --------
    >>> percent_change(50, 75)
    50.0
    >>> percent_change(100, 80)
    -20.0
    """
    return (new - old) / np.abs(old) * 100


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

    Examples
    --------
    >>> combine_all_csvs('/path/to/csvs', 'combined.csv')
    Wrote combined CSV to combined.csv
    <DataFrame object>
    """
    # list all files in directory
    csv_files = glob(os.path.join(path, f"{prefix}*.csv"))

    # concatenate CSVs
    pd_list = [pd.read_csv(f) for f in csv_files]
    df_res = pd.concat(pd_list,
                       ignore_index=True)  # TODO fix FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.

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
    execution_time = calculate_execution_time(start, stop)
    print(f"Execution time: {execution_time}")
    if return_string:
        return execution_time
    

def calculate_execution_time(start, stop):
    """
    Calculate the execution time between two given timestamps.

    Parameters
    ----------
    start : float
        The start timestamp.
    stop : float
        The stop timestamp.

    Returns
    -------
    str
        The execution time in the format "H:MM:SS".

    Examples
    --------
    >>> start = time.time()
    >>> # Some code to measure execution time
    >>> stop = time.time()
    >>> calculate_execution_time(start, stop)
    '0:00:05'

    """
    return str(datetime.timedelta(seconds=round(stop - start)))


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
    Unpickle the object stored in the given file path and return it. If the file path does not exist, it is interpreted as a glob pattern.

    Parameters
    ----------
    path : str
        The path to the file containing the pickled object.

    Returns
    -------
    object
        The unpickled object.

    """
    if os.path.exists(path):
        with open(path, "rb") as results_file:
            return _pickle.load(results_file)
    else:
        result_files = sorted(glob(path))
        if len(result_files) == 0:
            raise ValueError(f'No files matching {os.path.basename(path)} found in {os.path.dirname(path)}')
        elif len(result_files) > 1:
            raise ValueError(f'Multiple files found: {result_files}')
        else:
            with open(result_files[0], "rb") as results_file:
                return _pickle.load(results_file)


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
