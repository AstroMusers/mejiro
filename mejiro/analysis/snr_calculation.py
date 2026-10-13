import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def get_snr(exposure, snr_per_pixel_threshold=1):
    """Find the highest SNR among connected pixels above the per-pixel threshold.

    Pixels with undefined per-pixel SNR are excluded from region finding. The
    result is ``None`` when no valid pixel exceeds the threshold.
    """
    snr_array = get_snr_array(exposure)
    valid_pixels = np.isfinite(snr_array) & (snr_array > snr_per_pixel_threshold)
    masked_snr_array = np.ma.masked_where(~valid_pixels, snr_array)
    if masked_snr_array.mask.all():
        return None, masked_snr_array

    structure = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    )
    logger.debug(f'Using structure\n{structure}')

    labeled_array, num_regions = ndimage.label(masked_snr_array.filled(0), structure=structure)
    logger.debug(f'Identified {num_regions} region(s)')

    snrs = []
    for region_index in range(1, num_regions + 1):
        region = labeled_array == region_index
        source_counts = np.sum(exposure.source_data[region])
        total_counts = np.sum(exposure.data[region])
        snr = source_counts / np.sqrt(total_counts)
        snrs.append(snr)
        logger.debug(f'Region {region_index}: SNR = {snr}')

    return (np.max(snrs) if snrs else None), masked_snr_array


def get_snr_array(exposure):
    """Calculate per-pixel SNR without replacing undefined values.

    Pixels with zero or negative total counts naturally produce non-finite SNR
    values, which callers should mask rather than interpret as measurements.
    """
    _validate_exposure_for_snr_calculation(exposure)
    with np.errstate(divide='ignore', invalid='ignore'):
        return exposure.source_data / np.sqrt(exposure.data)


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
    if exposure.lens_data is None or exposure.source_data is None:
        raise ValueError('Exposure must have lens and source surface brightnesses calculated separately by setting pieces=True.')
    