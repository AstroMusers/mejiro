import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from package.plots import plot_util

# TODO use **kwargs for params like colorbar label, title, filepath, colorbar boolean, etc.
# TODO make sure the plot can be shown in a nb as well as saved


def percentile(array, title='', cmap='binary', percentile=98, colorbar=False):
    percentile = np.percentile(array, 98)
    vmin = -0.25 * percentile
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=percentile)

    plt.imshow(array, cmap=cmap, norm=norm)
    plt.title(title)
    if colorbar:
        plt.colorbar()

    plt.show()


def log10(array, title='', cmap='viridis', colorbar=False):
    plt.imshow(np.log10(array), cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()

    plt.show()


def plot(array, title='', cmap='viridis', colorbar=False, colorbar_label=None):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    if colorbar:
        cbar = plt.colorbar()
        if colorbar_label:
            cbar.set_label(colorbar_label)

    plt.show()
