import random

import astropy.io.fits as pyfits
from webbpsf import roman


def get_kwargs_psf(kernel, oversample):
    return {
        'psf_type': 'PIXEL',
        'kernel_point_source': kernel,
        'point_source_supersampling_factor': oversample
    }


def get_random_psf_kernel(band, oversample=5, save=None):
    wfi = get_instrument(band)
    wfi.detector = get_random_detector(wfi)
    wfi.detector_position = get_random_position()
    kernel = wfi.calc_psf(oversample=oversample)
    if save is not None:
        kernel.writeto(save, overwrite=True)
    return kernel


def print_header(filepath):
    from pprint import pprint
    header = pyfits.getheader(filepath)
    pprint(header)


def load_psf(filepath):
    return pyfits.getdata(filepath)


def get_instrument(band):
    wfi = roman.WFI()
    wfi.filter = band.upper()
    return wfi


def get_random_position():
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels
    return random.randrange(4, 4092), random.randrange(4, 4092)


def get_random_detector(wfi):
    return random.choice(wfi.detector_list)
