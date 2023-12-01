from os import path

import matplotlib.pyplot as plt

from package.utils import util


# def __kwargs_handler(kwargs):
#     if 'colorbar' in kwargs:
#         if kwargs['colorbar']:
#             plt.colorbar()
#             if 'colorbar_label' in kwargs:
#                 if kwargs['colorbar_label']:
#                     # TODO this is horrible, fix this


def __savefig(filepath):
    if filepath is not None:
        # check if the specified directory exists; if not, create it
        file_dir = path.dirname(filepath)
        util.create_directory_if_not_exists(file_dir)

        # save figure
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
