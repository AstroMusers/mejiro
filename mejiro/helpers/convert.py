import numpy as np


def get_zeropoint_magnitude(wavelength, response, effective_area=4.5 * 1e4):
    '''
    see Section 6.1 of [this paper](https://www.aanda.org/articles/aa/full_html/2022/06/aa42897-21/aa42897-21.html) by the Euclid collaboration for explanation of this function

    Roman's collecting area (4.5 m^2) retrieved 16 August 2024 from https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/introduction-to-wfi/wfi-quick-reference
    '''
    # effective area in cm^2

    # assert that wavelength values are evenly spaced
    assert np.allclose(np.diff(wavelength), np.diff(wavelength)[0])

    dv = np.diff(wavelength)[0]
    integral = 0
    for wl, resp in zip(wavelength, response):
        integral += (dv * (1 / wl) * resp)

    return 8.9 + (2.5 * np.log10(((effective_area * 1e-23) / (6.602 * 1e-27)) * integral))


def mjy_to_counts(array, band):
    conversion_factor = get_mjy_to_counts_factor(band)
    return array * conversion_factor


def counts_to_mjy(array, band):
    conversion_factor = get_counts_to_mjy_factor(band)
    return array * conversion_factor


def get_mjy_to_counts_factor(band):
    mjy_to_counts = {
        'f106': 4765.629510460321,
        'f129': 4009.2786141694673,
        'f158': 3227.044734375456,
        'f184': 1969.1718210052677
    }

    return mjy_to_counts[band]


def get_counts_to_mjy_factor(band):
    counts_to_mjy = {
        'f106': 0.00020983586697309335,
        'f129': 0.0002494214286993753,
        'f158': 0.0003098810466888475,
        'f184': 0.000507827701642357
    }

    return counts_to_mjy[band]
