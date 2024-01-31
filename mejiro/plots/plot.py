import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from mejiro.plots import plot_util


# TODO use **kwargs for params like colorbar label, title, filepath, colorbar boolean, etc.
# TODO make sure the plot can be shown in a nb as well as saved


def percentile(array, title='', cmap='binary', percentile=98, colorbar=False):
    norm = plot_util.percentile_norm(array, percentile)

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
    plt.imshow(plot_util.asinh(array), cmap=cmap)
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


def rgb_plot_grid(array_list, side, titles=False):
    f, ax = plt.subplots(side, side, figsize=(20, 20), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    i = 0
    for x in range(side):
        for y in range(side):
            ax[x][y].imshow(array_list[i])
            ax[x][y].set_axis_off()
            if titles:
                ax[x][y].set_title(f'{i}', color='red')
            i += 1

    plt.show()


def _plot_grid(array_list, side, cmap='viridis', log10=True, title='', save=None, colorbar=False):
    array_list = array_list[:side ** 2]
    if colorbar:
        vmin, vmax = plot_util.get_min_max(array_list)

    f, ax = plt.subplots(nrows=side, ncols=side, figsize=(20, 20), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    for i, image in enumerate(array_list):
        if log10:
            image = np.log10(image)
        if colorbar:
            ax[i // side, i % side].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax[i // side, i % side].imshow(image, cmap=cmap)
        ax[i // side, i % side].get_xaxis().set_visible(False)
        ax[i // side, i % side].get_yaxis().set_visible(False)

    plt.suptitle(title)

    # TODO fix
    # if colorbar:
    #     plt.colorbar()

    if save is not None:
        plt.savefig(save)

    plt.show()


def plot_grid(array_list, side, cmap='viridis', log10=True, title='', save=None, colorbar=False, colorbar_label=None):
    fig = plt.figure(figsize=(20, 20))

    cbar_kwargs = {
        'cbar_location': 'right',
        'cbar_mode': 'single',
        'cbar_pad': '2%',
        'cbar_size': '5%'
    }

    if colorbar:
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(side, side),
            axes_pad=0.04,
            label_mode='all',
            share_all=True,
            **cbar_kwargs)
    else:
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(side, side),
            axes_pad=0.04,
            label_mode='all',
            share_all=True)

    for i, ax in enumerate(grid):
        if log10:
            array_list[i] = np.log10(array_list[i])
        im = ax.imshow(array_list[i], cmap=cmap)

    if colorbar:
        cbar = grid.cbar_axes[0].colorbar(im)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label, rotation=90)
        for cax in grid.cbar_axes:
            cax.tick_params(labeltop=False)

    grid.axes_llc.set(xticks=[], yticks=[])

    plt.suptitle(title)

    if save is not None:
        plt.savefig(save)

    plt.show()


def plot_list(array_list, cmap='viridis', title_list=None, colorbar=False):
    f, ax = plt.subplots(nrows=1, ncols=len(array_list), figsize=(len(array_list) * 4, 4),
                         gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    if colorbar:
        vmin, vmax = plot_util.get_min_max(array_list)

    for i, array in enumerate(array_list):
        if colorbar:
            ax[i].imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax[i].imshow(array, cmap=cmap)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        if title_list is not None:
            ax[i].set_title(title_list[i])

    # TODO fix
    # if colorbar:
    #     plt.colorbar()

    plt.show()


def log10_list(array_list, cmap='viridis'):
    f, ax = plt.subplots(nrows=1, ncols=len(array_list), figsize=(len(array_list) * 4, 4),
                         gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    for i, array in enumerate(array_list):
        ax[i].imshow(np.log10(array), cmap='viridis')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    plt.show()
