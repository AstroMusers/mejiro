import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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


def arcsinh(array, title='', cmap='viridis', colorbar=False):
    array = np.arcsinh(array)
    array -= np.amin(array)
    array /= np.amax(array)

    plt.imshow(array, cmap=cmap)
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


def plot_grid(array_list, side, cmap='viridis'):
    f, ax = plt.subplots(nrows=side, ncols=side, figsize=(20, 20), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    for i, image in enumerate(array_list):
        ax[i // side, i % side].imshow(np.log10(image), cmap=cmap)
        ax[i // side, i % side].get_xaxis().set_visible(False)
        ax[i // side, i % side].get_yaxis().set_visible(False)

    plt.show()


def plot_list(array_list, cmap='viridis'):
    f, ax = plt.subplots(nrows=1, ncols=len(array_list), figsize=(len(array_list) * 4, 4),
                         gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    for i, array in enumerate(array_list):
        ax[i].imshow(array, cmap='viridis')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    plt.show()


def log10_list(array_list, cmap='viridis'):
    f, ax = plt.subplots(nrows=1, ncols=len(array_list), figsize=(len(array_list) * 4, 4),
                         gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    for i, array in enumerate(array_list):
        ax[i].imshow(np.log10(array), cmap='viridis')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    plt.show()
