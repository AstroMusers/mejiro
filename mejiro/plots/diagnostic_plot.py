import logging
from os import path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.fft import fft2

from mejiro.lenses import lens_util
from mejiro.utils import util

logger = logging.getLogger(__name__)


def get_v(array_list):
    """Return the largest absolute extremum across a list of arrays.

    For each array the greater of ``|min|`` and ``|max|`` is computed; the
    overall maximum of those per-array values is returned. Used to set a
    symmetric color scale (``vmin=-v, vmax=v``) for residual panels.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        Arrays to inspect.

    Returns
    -------
    float
        Largest absolute value found across all arrays.
    """
    max_list = []
    for array in array_list:
        abs_min, abs_max = abs(np.min(array)), abs(np.max(array))
        max_list.append(np.max([abs_min, abs_max]))
    return np.max(max_list)


def get_norm(array_list, linear_width):
    """Build a symmetric ``AsinhNorm`` that spans the full range of a list of arrays.

    Finds the largest absolute value across all arrays and uses it as the
    symmetric ``vmin``/``vmax``, with an arcsinh stretch controlled by
    *linear_width*.

    Parameters
    ----------
    array_list : list of numpy.ndarray
        Arrays whose combined range determines the normalization limits.
    linear_width : float
        The ``linear_width`` parameter passed to ``matplotlib.colors.AsinhNorm``;
        controls the transition between linear and logarithmic scaling near zero.

    Returns
    -------
    matplotlib.colors.AsinhNorm
        Normalization instance with ``vmin=-limit`` and ``vmax=limit``.
    """
    min_list, max_list = [], []
    for array in array_list:
        min_list.append(abs(np.min(array)))
        max_list.append(abs(np.max(array)))
    abs_min, abs_max = abs(np.min(min_list)), abs(np.max(max_list))
    limit = np.max([abs_min, abs_max])
    return colors.AsinhNorm(linear_width=linear_width, vmin=-limit, vmax=limit)


def __savefig(filepath):
    """Save the current figure to *filepath* and close it.

    Creates any missing parent directories before saving.  Does nothing if
    *filepath* is ``None``.

    Parameters
    ----------
    filepath : str or None
        Destination path for the saved figure.  If ``None``, the function
        returns immediately without saving.

    Returns
    -------
    None
    """
    if filepath is not None:
        file_dir = path.dirname(filepath)
        util.create_directory_if_not_exists(file_dir)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()


def snr_plot(labeled_array, strong_lens, total, lens, source, noise, snr_array, masked_snr_array, snr_list, debug_dir):
    """Save a 2×3 diagnostic panel showing SNR-related image components.

    Panels (row, col):
    - (0,0) Total image, log10-scaled
    - (0,1) Lens-light image
    - (0,2) Source image
    - (1,0) Noise image
    - (1,1) Labeled array (connected-component regions)
    - (1,2) Masked SNR array

    The figure is saved to ``{debug_dir}/snr/snr_check_{id(total)}.png``.

    Parameters
    ----------
    labeled_array : numpy.ndarray
        Integer-labeled array of detected source regions.
    strong_lens : StrongLens
        Lens object used to retrieve F129 magnitudes for panel titles.
    total : numpy.ndarray
        Total (lens + source + noise) image array.
    lens : numpy.ndarray
        Lens-light-only image array.
    source : numpy.ndarray
        Source-light-only image array.
    noise : numpy.ndarray
        Noise image array.
    snr_array : numpy.ndarray
        Per-pixel SNR array (not currently displayed but kept for signature
        compatibility).
    masked_snr_array : numpy.ndarray
        SNR array after masking non-source regions.
    snr_list : list of float
        List of per-region SNR values; the maximum is shown in the super-title.
    debug_dir : str
        Root directory under which the ``snr/`` subdirectory is expected.

    Returns
    -------
    None
    """
    _, ax = plt.subplots(2, 3, figsize=(12, 8))

    lens_mag = strong_lens.lens_mags['F129']
    source_mag = strong_lens.source_mags['F129']

    im00 = ax[0][0].imshow(np.log10(total))  # , vmin=vmin, vmax=vmax
    plt.colorbar(im00, ax=ax[0][0])
    ax[0][0].set_title('Total Image (log10)')

    im01 = ax[0][1].imshow(lens)
    plt.colorbar(im01, ax=ax[0][1])
    ax[0][1].set_title('Lens (' + r'$m_\textrm{F129}=$' + f'{lens_mag:.2f})')

    im02 = ax[0][2].imshow(source)
    plt.colorbar(im02, ax=ax[0][2])
    ax[0][2].set_title('Source (' + r'$m_\textrm{F129}=$' + f'{source_mag:.2f})')

    im10 = ax[1][0].imshow(noise)
    plt.colorbar(im10, ax=ax[1][0])
    ax[1][0].set_title('Noise')

    im11 = ax[1][1].imshow(labeled_array)
    # for k, region in enumerate(indices_list):
    #     for i, j in region:
    #         ax[1][1].plot(j, i, 'ro', markersize=1, color=f'C{k}')
    plt.colorbar(im11, ax=ax[1][1])
    ax[1][1].set_title('Labeled Array')

    im12 = ax[1][2].imshow(masked_snr_array)
    plt.colorbar(im12, ax=ax[1][2])
    ax[1][2].set_title('Masked SNR Array')

    plt.suptitle(f'SNR: {np.max(snr_list)}, z_l={strong_lens.z_lens:.2f}, z_s={strong_lens.z_source:.2f}')
    try:
        plt.savefig(f'{debug_dir}/snr/snr_check_{id(total)}.png')
        plt.close()
    except Exception as e:
        logger.warning(f'Could not save SNR plot: {e}')


def power_spectrum_check(array_list, lenses, titles, save_path, oversampled):
    """Save a 2×4 panel comparing images and their residuals against a reference.

    The top row shows log10-scaled images; the bottom row shows residuals
    relative to the last image (``array_list[3]``) on a symmetric ``bwr``
    scale.  Halos with mass > 1e8 M☉ are overlaid as open circles on the
    residual panels.

    Parameters
    ----------
    array_list : list of numpy.ndarray or list of object
        Four image arrays, or objects with an ``.array`` attribute.  The last
        element (index 3) is used as the reference for residuals.
    lenses : list
        Lens objects aligned with *array_list*; each may have a
        ``realization`` attribute containing ``halos``.
    titles : list of str
        Panel titles for the top row, one per array.
    save_path : str
        File path at which to save the figure.
    oversampled : bool
        If ``True``, halo coordinates are computed on a 5× oversampled grid
        (45×5 pixels, 0.11/5 arcsec/pixel); otherwise on the native grid
        (45 pixels, 0.11 arcsec/pixel).

    Returns
    -------
    None
    """
    if type(array_list[0]) is not np.ndarray:
        array_list = [i.array for i in array_list]

    f, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i, array in enumerate(array_list):
        axis = ax[0][i].imshow(np.log10(array))
        ax[0][i].set_title(titles[i])
        ax[0][i].axis('off')

    cbar = f.colorbar(axis, ax=ax[0])
    cbar.set_label('log(Counts)', rotation=90)

    res_array = [array_list[3] - array_list[i] for i in range(4)]
    v = get_v(res_array)
    for i in range(4):
        axis = ax[1][i].imshow(array_list[3] - array_list[i], cmap='bwr', vmin=-v, vmax=v)
        ax[1][i].set_axis_off()

    cbar = f.colorbar(axis, ax=ax[1])
    cbar.set_label('Counts', rotation=90)

    for i, lens in enumerate(lenses):
        realization = lens.realization
        if realization is not None:
            for halo in realization.halos:
                if halo.mass > 1e8:
                    if oversampled:
                        coords = lens_util.get_coords(45 * 5, delta_pix=0.11 / 5)
                    else:
                        coords = lens_util.get_coords(45, delta_pix=0.11)
                    ax[1][i].scatter(*coords.map_coord2pix(halo.x, halo.y), s=100, facecolors='none',
                                     edgecolors='black')

    plt.savefig(save_path)
    plt.close()


def residual_compare(ax, array_list, linear_width, title_list=None):
    """Populate a row of axes with residual images relative to the last array.

    Each panel shows ``array_list[-1] - array_list[i]`` using a shared
    ``AsinhNorm`` color scale derived from the full range of *array_list*.

    Parameters
    ----------
    ax : array-like of matplotlib.axes.Axes
        Pre-created axes to fill, one per array in *array_list*.
    array_list : list of numpy.ndarray
        Images to compare; the last element is the reference.
    linear_width : float
        Passed to `get_norm` to control the arcsinh linear-to-log transition.
    title_list : list of str or None, optional
        Per-panel titles aligned with *array_list*.  Defaults to ``None``.

    Returns
    -------
    matplotlib.image.AxesImage
        The ``AxesImage`` object from the last panel, suitable for adding a
        colorbar.
    """
    norm = get_norm(array_list, linear_width)

    last_array = array_list[:-1]

    for i, array in enumerate(array_list):
        axis = ax[i].imshow(last_array - array, cmap='bwr', norm=norm)
        ax[i].set_title(title_list[i])
        ax[i].set_axis_off()

    return axis


def fft(filepath, title, array):
    """Display and optionally save the 2-D FFT magnitude of an array.

    Computes the 2-D FFT, displays its absolute value with a log color scale,
    saves to *filepath* if provided, then shows the figure.

    Parameters
    ----------
    filepath : str or None
        Destination path for saving the figure.  Passed to ``__savefig``; if
        ``None``, the figure is displayed but not saved.
    title : str
        Plot title.
    array : numpy.ndarray
        2-D input array on which the FFT is computed.

    Returns
    -------
    None
    """
    fft = fft2(array)
    plt.matshow(np.abs(fft), norm=colors.LogNorm())
    plt.title(title)
    plt.colorbar()
    __savefig(filepath)
    plt.show()


def residual(array1, array2, title='', normalization=1):
    """Display the normalized residual between two arrays.

    Computes ``(array1 - array2) / normalization`` and renders it on a
    symmetric ``bwr`` color scale using an ``AsinhNorm`` whose linear width
    is set to ``|mean + 3*std|`` of the residual.

    Parameters
    ----------
    array1 : numpy.ndarray
        First (e.g. observed) image array.
    array2 : numpy.ndarray
        Second (e.g. model) image array with the same shape as *array1*.
    title : str, optional
        Plot title.  Defaults to ``''``.
    normalization : float, optional
        Scalar divisor applied to the raw difference before display.
        Defaults to ``1`` (no normalization).

    Returns
    -------
    None
    """
    residual = (array1 - array2) / normalization
    abs_min, abs_max = abs(np.min(residual)), abs(np.max(residual))
    limit = np.max([abs_min, abs_max])
    linear_width = np.abs(np.mean(residual) + (3 * np.std(residual)))

    fig, ax = plt.subplots()
    im = ax.imshow(residual, cmap='bwr', norm=colors.AsinhNorm(linear_width=linear_width, vmin=-limit, vmax=limit))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    plt.show()


def execution_time_scatter(execution_times, title=''):
    """Display a scatter plot of execution times by index.

    Parameters
    ----------
    execution_times : array-like of float
        Sequence of per-task execution times to plot.
    title : str, optional
        Plot title.  Defaults to ``''``.

    Returns
    -------
    None
    """
    plt.scatter(np.arange(0, len(execution_times)), execution_times)
    plt.title(title)

    plt.show()


def execution_time_hist(execution_times, title=''):
    """Display a histogram of execution times.

    Parameters
    ----------
    execution_times : array-like of float
        Sequence of per-task execution times to bin.
    title : str, optional
        Plot title.  Defaults to ``''``.

    Returns
    -------
    None
    """
    plt.hist(execution_times)
    plt.title(title)

    plt.show()
