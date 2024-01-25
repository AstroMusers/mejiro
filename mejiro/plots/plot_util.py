import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from os import path

from mejiro.utils import util


def normalize(array):
    return array / np.linalg.norm(array)


def get_residual_list(array_list):
    last_array = array_list[-1]
    residual_list = [(last_array - i) for i in array_list]
    return residual_list[:-1]


def get_filenames(filepath_list):
    return [path.basename(i) for i in filepath_list]


def asinh(array):
    array = np.arcsinh(array)
    array -= np.amin(array)
    array /= np.amax(array)
    return array


def percentile_norm(array, percentile):
    percentile = np.percentile(array, percentile)
    vmin = -0.25 * percentile
    return colors.Normalize(vmin=vmin, vmax=percentile)


def get_norm(array_list, linear_width):
    min_list, max_list = [], []
    for array in array_list:
        min_list.append(abs(np.min(array)))
        max_list.append(abs(np.max(array)))
    abs_min, abs_max = abs(np.min(min_list)), abs(np.max(max_list))
    limit = np.max([abs_min, abs_max])
    return colors.AsinhNorm(linear_width=linear_width, vmin=-limit, vmax=limit)


def get_limit(array):
    abs_min, abs_max = abs(np.min(array)), abs(np.max(array))
    return np.max([abs_min, abs_max])


def get_v(array_list):
    max_list = []
    for array in array_list:
        abs_min, abs_max = abs(np.min(array)), abs(np.max(array))
        max_list.append(np.max([abs_min, abs_max]))
    return np.max(max_list)

def set_v(array_list):
    limit = get_v(array_list)
    return {'vmin': -limit,
            'vmax': limit}


def get_min_max(array_list):
    min_list, max_list = [], []
    for array in array_list:
        min_list.append(np.min(array))
        max_list.append(np.max(array))
    return np.min(min_list), np.max(max_list)


def get_linear_width(array):
    return np.abs(np.mean(array) + (3 * np.std(array)))


def __savefig(filepath):
    if filepath is not None:
        # check if the specified directory exists; if not, create it
        file_dir = path.dirname(filepath)
        util.create_directory_if_not_exists(file_dir)

        # save figure
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
