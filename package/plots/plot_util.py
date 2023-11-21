import matplotlib
import matplotlib.pyplot as plt
from os import path

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

from package.utils import util

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'


def __savefig(filepath):
    if filepath is not None:
        # check if the specified directory exists; if not, create it
        file_dir = path.dirname(filepath)
        util.create_directory_if_not_exists(file_dir)

        # save figure
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        