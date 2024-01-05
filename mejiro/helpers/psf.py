import random

import astropy.io.fits as pyfits
from webbpsf import roman


def get_kwargs_psf(kernel, oversample):
    return {
        'psf_type': 'PIXEL',
        'kernel_point_source': kernel,
        'point_source_supersampling_factor': oversample
    }


def get_kernel_from_calc_psf(calc_psf):
    return calc_psf['DET_SAMP'].data


def get_random_psf_kernel(band, oversample=5, save=None, suppress_output=False):
    wfi = get_instrument(band)
    wfi.detector = get_random_detector(wfi, suppress_output)
    wfi.detector_position = get_random_position(suppress_output)
    psf = wfi.calc_psf(oversample=oversample)
    if save is not None:
        psf.writeto(save, overwrite=True)
    return psf['DET_SAMP'].data


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


def get_random_position(suppress_output=False):
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels
    x, y = random.randrange(4, 4092), random.randrange(4, 4092)
    if not suppress_output:
        print(f'Detector position: {x}, {y}')
    return x, y


def get_random_detector(wfi, suppress_output=False):
    detector = random.choice(wfi.detector_list)
    if not suppress_output:
        print(f'Detector: {detector}')
    return detector
