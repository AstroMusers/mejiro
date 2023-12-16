from os import path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from mejiro.utils import util


# def __kwargs_handler(kwargs):
#     if 'colorbar' in kwargs:
#         if kwargs['colorbar']:
#             plt.colorbar()
#             if 'colorbar_label' in kwargs:
#                 if kwargs['colorbar_label']:
#                     # TODO this is horrible, fix this


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
