import datetime
import os
import pickle as _pickle
import shutil
import pandas as pd
from collections import ChainMap
from glob import glob

import numpy as np
from omegaconf import OmegaConf


def combine_all_csvs(path, filename):
    # list all files in directory
    csv_files = glob(os.path.join(path, '*.csv'))

    # concatenate CSVs
    pd_list = [pd.read_csv(os.path.join(path, f)) for f in csv_files]
    df_res = pd.concat(pd_list, ignore_index=True)

    # save as combined CSV
    df_res.to_csv(filename)
    print(f'Wrote combined CSV to {filename}')

    # return as DataFrame
    return df_res


def check_negative_values(array):
    # takes an array or a list of arrays
    if isinstance(array, list):
        for a in array:
            if np.any(a < 0):
                return True
    else:
        return np.any(array < 0)


def replace_negatives_with_zeros(array):
    return np.where(array < 0, 0, array)


def resize_with_pixels_centered(array, oversample_factor):
    if oversample_factor % 2 == 0:
        raise Exception('Oversampling factor must be odd')

    x, y = array.shape
    if x != y:
        raise Exception('Array must be square')

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
    if array.shape == shape:
        return array

    y_out, x_out = shape
    tuple = array.shape
    y, x = tuple[0], tuple[1]
    x_start = (x // 2) - (x_out // 2)
    y_start = (y // 2) - (y_out // 2)
    return array[y_start:y_start + y_out, x_start:x_start + x_out]


def hydra_to_dict(config):
    container = OmegaConf.to_container(config, resolve=True)
    return dict(ChainMap(*container))


def print_execution_time(start, stop):
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def pickle(path, thing):
    with open(path, 'ab') as results_file:
        _pickle.dump(thing, results_file)


def unpickle(path):
    with open(path, 'rb') as results_file:
        result = _pickle.load(results_file)
    return result


def unpickle_all(dir_path, prefix='', limit=None):
    file_list = glob(dir_path + f'/{prefix}*')
    sorted_list = sorted(file_list)
    if limit is not None:
        return [unpickle(i) for i in sorted_list[:limit] if os.path.isfile(i)]
    else:
        return [unpickle(i) for i in sorted_list if os.path.isfile(i)]


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_directory(path):
    for i in glob(path + '/*'):
        if os.path.isfile(i):
            os.remove(i)
        else:
            shutil.rmtree(i)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def scientific_notation_string(input):
    return '{:.2e}'.format(input)


def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
