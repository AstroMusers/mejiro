import numpy as np


def get_snr(exposure, snr_per_pixel_threshold=1, verbose=False):
    from scipy import ndimage

    snr_array = get_snr_array(exposure)
    masked_snr_array = np.ma.masked_where(snr_array <= snr_per_pixel_threshold, snr_array)

    structure = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])
    if verbose: print(f'Using structure: {structure}')

    labeled_array, num_regions = ndimage.label(masked_snr_array.filled(0), structure=structure)
    if verbose: print(f'Identified {num_regions} regions')

    # calculate the SNR for each region
    snrs = []
    for i in range(1, num_regions + 1):
        source_counts = np.sum(exposure.source_exposure[labeled_array == i])
        total_counts = np.sum(exposure.exposure[labeled_array == i])
        snr = source_counts / np.sqrt(total_counts)
        snrs.append(snr)
        if verbose: print(f'Region {i}: SNR = {snr}')

    # return the maximum SNR
    return np.max(snrs) if snrs else None


def get_snr_array(exposure):
    _validate_exposure_for_snr_calculation(exposure)
    return exposure.source_exposure / np.sqrt(exposure.exposure)


def _validate_exposure_for_snr_calculation(exposure):
    # check that the exposure has calculated the source and lens surface brightnesses separately
    if exposure.lens_exposure is None or exposure.source_exposure is None:
        raise ValueError('Exposure must have lens and source surface brightnesses calculated separately')
    