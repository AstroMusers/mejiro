import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import ImageGrid


# TODO use **kwargs for params like colorbar label, title, filepath, colorbar boolean, etc.
# TODO make sure the plot can be shown in a nb as well as saved


def percentile_norm(array, percentile):
    """Build a linear ``Normalize`` instance clipped to a given percentile.

    Sets ``vmax`` to the requested percentile value and ``vmin`` to
    ``-0.25 * vmax``, which slightly extends the low end below zero to
    preserve faint negative features.

    Parameters
    ----------
    array : numpy.ndarray
        Input array from which the percentile is computed.
    percentile : float
        Percentile (0–100) used to set ``vmax``.

    Returns
    -------
    matplotlib.colors.Normalize
        Normalization instance with ``vmin=-0.25*p`` and ``vmax=p`` where
        *p* is the computed percentile value.
    """
    percentile = np.percentile(array, percentile)
    vmin = -0.25 * percentile
    return colors.Normalize(vmin=vmin, vmax=percentile)


def asinh(array):
    """Apply an arcsinh stretch and rescale values to ``[0, 1]``.

    Computes ``arcsinh(array)``, shifts so the minimum is zero, then divides
    by the maximum, producing a display-ready array with values in ``[0, 1]``.

    Parameters
    ----------
    array : numpy.ndarray
        Input pixel array (e.g. an image cutout).

    Returns
    -------
    numpy.ndarray
        Arcsinh-stretched and min-max-normalized array with the same shape as
        *array*, values in ``[0, 1]``.
    """
    array = np.arcsinh(array)
    array -= np.amin(array)
    array /= np.amax(array)
    return array


def get_min_max(array_list):
    """Return the global minimum and maximum across a list of arrays.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        Arrays to search.

    Returns
    -------
    vmin : float
        Smallest value found across all arrays.
    vmax : float
        Largest value found across all arrays.
    """
    min_list, max_list = [], []
    for array in array_list:
        min_list.append(np.min(array))
        max_list.append(np.max(array))
    return np.min(min_list), np.max(max_list)


def percentile(array, title='', cmap='binary', percentile=98, colorbar=False):
    """Display an image normalized to a given percentile.

    Parameters
    ----------
    array : numpy.ndarray
        2-D image array to display.
    title : str, optional
        Plot title.  Defaults to ``''``.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'binary'``.
    percentile : float, optional
        Percentile (0–100) passed to `percentile_norm` to set ``vmax``.
        Defaults to ``98``.
    colorbar : bool, optional
        Whether to add a colorbar.  Defaults to ``False``.

    Returns
    -------
    None
    """
    norm = percentile_norm(array, percentile)

    plt.imshow(array, cmap=cmap, norm=norm)
    plt.title(title)
    if colorbar:
        plt.colorbar()

    plt.show()


def log10(array, title='', cmap='viridis', colorbar=False):
    """Display an image with log10 scaling normalized to its maximum.

    The array is divided by its maximum before taking ``log10``, so pixel
    values span ``(-inf, 0]`` with the brightest pixel at zero.

    Parameters
    ----------
    array : numpy.ndarray
        2-D image array to display.  All values must be positive.
    title : str, optional
        Plot title.  Defaults to ``''``.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'viridis'``.
    colorbar : bool, optional
        Whether to add a colorbar.  Defaults to ``False``.

    Returns
    -------
    None
    """
    plt.imshow(np.log10(array / np.max(array)), cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()

    plt.show()


def arcsinh(array, title='', cmap='viridis', colorbar=False):
    """Display an image with arcsinh stretching rescaled to ``[0, 1]``.

    Parameters
    ----------
    array : numpy.ndarray
        2-D image array to display.
    title : str, optional
        Plot title.  Defaults to ``''``.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'viridis'``.
    colorbar : bool, optional
        Whether to add a colorbar.  Defaults to ``False``.

    Returns
    -------
    None
    """
    plt.imshow(asinh(array), cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()

    plt.show()


def plot(array, title='', cmap='viridis', colorbar=False, colorbar_label=None):
    """Display an image with no additional scaling.

    Parameters
    ----------
    array : numpy.ndarray
        2-D image array to display.
    title : str, optional
        Plot title.  Defaults to ``''``.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'viridis'``.
    colorbar : bool, optional
        Whether to add a colorbar.  Defaults to ``False``.
    colorbar_label : str or None, optional
        Label for the colorbar axis.  Only used when *colorbar* is ``True``.
        Defaults to ``None``.

    Returns
    -------
    None
    """
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    if colorbar:
        cbar = plt.colorbar()
        if colorbar_label:
            cbar.set_label(colorbar_label)

    plt.show()


def rgb_plot_grid(array_list, side, titles=None, save=None):
    """Display a square grid of RGB images.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        List of ``side*side`` RGB image arrays (shape ``(H, W, 3)``).
    side : int
        Number of rows and columns in the grid.
    titles : list of str or None, optional
        Per-panel titles.  When provided, ``constrained_layout`` is enabled.
        Defaults to ``None``.
    save : str or None, optional
        File path to save the figure.  If ``None`` (default), the figure is
        not saved.

    Returns
    -------
    None
    """
    subplot_kwargs = {
        'nrows': side, 'ncols': side, 'figsize': (20, 20)
    }
    if titles is not None:
        subplot_kwargs['constrained_layout'] = True
    else:
        subplot_kwargs['gridspec_kw'] = {'hspace': 0.02, 'wspace': 0.02}

    f, ax = plt.subplots(**subplot_kwargs)

    i = 0
    for x in range(side):
        for y in range(side):
            ax[x][y].imshow(array_list[i])
            ax[x][y].set_axis_off()
            if titles:
                ax[x][y].set_title(f'{titles[i]}')
            i += 1

    if save is not None:
        plt.savefig(save)

    plt.show()


def plot_grid(array_list, side, cmap='viridis', log10=True, title='', save=None, colorbar=False, colorbar_label=None):
    """Display a square grid of single-channel images using ``ImageGrid``.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        List of ``side*side`` 2-D image arrays.
    side : int
        Number of rows and columns in the grid.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'viridis'``.
    log10 : bool, optional
        If ``True`` (default), each image is log10-scaled before display.
    title : str, optional
        Super-title for the figure.  Defaults to ``''``.
    save : str or None, optional
        File path to save the figure.  If ``None`` (default), the figure is
        not saved.
    colorbar : bool, optional
        Whether to add a shared colorbar on the right.  Defaults to ``False``.
    colorbar_label : str or None, optional
        Label for the shared colorbar.  Only used when *colorbar* is ``True``.
        Defaults to ``None``.

    Returns
    -------
    None
    """
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
            im = ax.imshow(np.log10(array_list[i]), cmap=cmap)
        else:
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
    """Display a list of images as a single-row strip.

    When *colorbar* is ``True``, all panels share the same ``vmin``/``vmax``
    derived from the global minimum and maximum across all arrays.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        Ordered list of 2-D image arrays to display.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'viridis'``.
    title_list : list of str or None, optional
        Per-panel titles.  Defaults to ``None``.
    colorbar : bool, optional
        Whether to use a shared color scale across all panels.
        Defaults to ``False``.

    Returns
    -------
    None
    """
    f, ax = plt.subplots(nrows=1, ncols=len(array_list), figsize=(len(array_list) * 4, 4),
                         gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    if colorbar:
        vmin, vmax = get_min_max(array_list)

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
    """Display a list of images as a single-row strip with log10 scaling.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        Ordered list of 2-D image arrays to display.  All values must be
        positive.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``'viridis'``.

    Returns
    -------
    None
    """
    f, ax = plt.subplots(nrows=1, ncols=len(array_list), figsize=(len(array_list) * 4, 4),
                         gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    for i, array in enumerate(array_list):
        ax[i].imshow(np.log10(array), cmap='viridis')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    plt.show()
