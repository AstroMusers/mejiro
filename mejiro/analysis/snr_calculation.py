import numpy as np
from scipy import ndimage


def get_snr(exposure, snr_per_pixel_threshold=1, verbose=False):
    """
    Calculate the signal-to-noise ratio (SNR) given an exposure, using the method of `Holloway et al. (2023) <https://doi.org/10.1093/mnras/stad2371>`_. First, the SNR per pixel is calculated: see `get_snr_array()`. Then, contiguous regions of pixels above the SNR per pixel threshold are identified. The SNR for each region is calculated in the following way:

    .. math::

        \\text{SNR}_\\text{region} = \\frac{\\sum\\limits_i N_{i,\\,S}}{\\sqrt{\\sum\\limits_i \\left(N_{i,\\,S} + N_{i,\\,L} + N_{i,\\,B} + N_{i,\\,N}\\right)}}

    where the summations are over the pixels that comprise the region, :math:`N_{i,\,S}` are the counts in pixel :math:`i` due to the source galaxy, :math:`N_{i,\,L}` are counts due to the lensing galaxy, :math:`N_{i,\,B}` are counts due to the sky background, and :math:`N_{i,\,N}` are counts due to detector noise. If multiple regions are formed, the SNR of the region with the highest SNR is taken to be the SNR of the system.

    Parameters
    ----------
    exposure : object
        Expected to have 'source_exposure' and 'exposure' attributes.
    snr_per_pixel_threshold : float, optional
        The minimum SNR per pixel required for a pixel to be included in a region (default is 1).
    verbose : bool, optional
        If True, prints detailed information about the processing steps (default is False).

    Returns
    -------
    max_snr : float or None
        The maximum SNR found among the regions above the threshold. Returns 1 if no pixels are above the threshold, or None if no regions are found.
    masked_snr_array : numpy.ma.MaskedArray
        The SNR array with pixels below the threshold masked.

    Notes
    -----
    - Regions are defined as contiguous pixels (using a cross-shaped connectivity) above the SNR threshold.
    - If no pixels exceed the threshold, the SNR returned is 1. Any system with no pixels above the threshold is undetectable. Typically, an SNR of 20 is required for a system to be detectable.
    """
    snr_array = get_snr_array(exposure)
    masked_snr_array = np.ma.masked_where(snr_array <= snr_per_pixel_threshold, snr_array)

    # if no pixels are above the threshold, return 1
    if masked_snr_array.mask.all():
        return 1, masked_snr_array

    structure = np.array(
                    [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])
    if verbose: print(f'Using structure: {structure}')

    labeled_array, num_regions = ndimage.label(masked_snr_array.filled(0), structure=structure)
    if verbose: print(f'Identified {num_regions} region(s)')

    # calculate the SNR for each region
    snrs = []
    for i in range(1, num_regions + 1):
        source_counts = np.sum(exposure.source_exposure[labeled_array == i])
        total_counts = np.sum(exposure.exposure[labeled_array == i])
        snr = source_counts / np.sqrt(total_counts)
        snrs.append(snr)
        if verbose: print(f'Region {i}: SNR = {snr}')

    # return the maximum SNR
    return np.max(snrs) if snrs else None, masked_snr_array


def get_snr_array(exposure):
    """
    Calculate the signal-to-noise ratio (SNR) per pixel for a given exposure: 

    .. math::

        \\frac{\\text{Source}}{\\sqrt{\\text{Source + Lens + Noise}}}

    Any NaN or infinite values (for example, if a pixel happens to have zero counts) are replaced with zero.

    Parameters
    ----------
    exposure : object
        Expected to have 'source_exposure' and 'exposure' attributes.

    Returns
    -------
    np.ndarray
        An array containing the SNR values, with NaN and infinite values replaced by zero.

    Raises
    ------
    ValueError
        If the exposure was not created with `pieces=True`.
    """
    _validate_exposure_for_snr_calculation(exposure)
    return np.nan_to_num(exposure.source_exposure / np.sqrt(exposure.exposure), nan=0, posinf=0, neginf=0)


def _validate_exposure_for_snr_calculation(exposure):
    """
    Validates that the given exposure has calculated the source and lens surface 
    brightnesses separately.

    Parameters
    ----------
    exposure : object

    Raises
    ------
    ValueError
        If either `lens_exposure` or `source_exposure` is None, indicating that the
        exposure does not have the required separate surface brightness calculations.
    """
    if exposure.lens_exposure is None or exposure.source_exposure is None:
        raise ValueError('Exposure must have lens and source surface brightnesses calculated separately by setting pieces=True.')
    