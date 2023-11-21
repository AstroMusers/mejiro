import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from matplotlib import rc

from package.plots import plot_util

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'


# def overplot_lens_params(filepath, title, array, cmap='viridis', colorbar=False, colorbar_label=None):

# TODO use **kwargs for params like colorbar label, title, filepath, colorbar boolean, etc.


def percentile(filepath, title, array, cmap='binary', percentile=98, colorbar=False):
    percentile = np.percentile(array, 98)
    vmin = -0.25 * percentile
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=percentile)

    plt.imshow(array, cmap=cmap, norm=norm)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    plot_util.__savefig(filepath)
    plt.show()



def log10(filepath, title, array, cmap='viridis', colorbar=False):
    plt.imshow(np.log10(array), cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    plot_util.__savefig(filepath)
    plt.show()


def plot(filepath, title, array, cmap='viridis', colorbar=False, colorbar_label=None):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    if colorbar:
        cbar = plt.colorbar()
        if colorbar_label:
            cbar.set_label(colorbar_label)
    plot_util.__savefig(filepath)
    plt.show()
