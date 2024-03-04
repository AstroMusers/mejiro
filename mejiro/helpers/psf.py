import os
import random
from glob import glob

import astropy.io.fits as pyfits
import galsim
from galsim import roman
from tqdm import tqdm
from webbpsf.roman import WFI

from mejiro.helpers import gs


def get_instrument(band):
    wfi = WFI()
    wfi.filter = band.upper()
    return wfi


def get_random_detector(suppress_output=False):
    detector = random.choice(WFI().detector_list)
    if not suppress_output:
        print(f'Detector: {detector}')
    return detector


def get_random_detector_pos(input_size, suppress_output=False):
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels. We're
    # adjusting inwards by the input_size because we want to make sure that the entire image fits on the detector,
    # even before final cropping to remove any edge effects
    min_pixel = 4 + input_size
    max_pixel = 4092 - input_size

    x, y = random.randrange(min_pixel, max_pixel), random.randrange(min_pixel, max_pixel)

    if not suppress_output:
        print(f'Detector position: {x}, {y}')
    return galsim.PositionD(x, y)


def get_webbpsf_psf(band, detector, detector_position, oversample):
    # detector might be int or string
    if type(detector) == int:
        detector = f'SCA{str(detector).zfill(2)}'

    # set PSF parameters
    wfi = get_instrument(band)
    wfi.detector = detector  # WebbPSF expects 'SCA01', 'SCA02', etc.
    wfi.detector_position = detector_position

    # generate PSF in WebbPSF
    psf = wfi.calc_psf(oversample=oversample)

    # import PSF to GalSim
    oversampled_pixel_scale = wfi.pixelscale / oversample
    psf_image = galsim.Image(psf[0].data, scale=oversampled_pixel_scale)
    return galsim.InterpolatedImage(psf_image)


def get_galsim_psf(band, detector, detector_position, pupil_bin=1):
    return roman.getPSF(detector,
                        SCA_pos=galsim.PositionD(*detector_position),
                        bandpass=None,
                        wavelength=gs.get_bandpass(band),
                        pupil_bin=pupil_bin)


def get_kwargs_psf(kernel, oversample):
    return {
        'psf_type': 'PIXEL',
        'kernel_point_source': kernel,
        'point_source_supersampling_factor': oversample
    }


def get_psf_kernel(band, detector, detector_position, oversample=5, save=None):
    wfi = get_instrument(band)
    wfi.detector = detector
    wfi.detector_position = detector_position
    psf = wfi.calc_psf(oversample=oversample)
    if save is not None:
        psf.writeto(save, overwrite=True)
    return psf[0].data


def get_random_psf_kernel(band, oversample=5, save=None, suppress_output=False):
    wfi = get_instrument(band)
    wfi.detector = get_random_detector(suppress_output)
    wfi.detector_position = get_random_detector_pos(100, suppress_output)  # TODO refactor so input_size is meaningful
    psf = wfi.calc_psf(oversample=oversample)
    if save is not None:
        psf.writeto(save, overwrite=True)
    return psf[0].data


def print_header(filepath):
    from pprint import pprint
    header = pyfits.getheader(filepath)
    pprint(header)


def load_psf(filepath):
    return pyfits.getdata(filepath)


def load_default_psf(dir, band, oversample):
    filepath = os.path.join(dir, f'webbpsf_sca01_center_{band.lower()}_{oversample}.fits')
    return load_psf(filepath)


def _get_pandeia_psf_options():
    return {'source_offset_r': 0.,
            'source_offset_theta': 0.,
            'pupil_shift_x': 0.,
            'pupil_shift_y': 0.,
            'output_mode': 'oversampled',
            'jitter': 'gaussian',
            'jitter_sigma': 0.012}


def get_pandeia_psf_dir():
    pandeia_dir = os.environ['pandeia_refdata']
    return os.path.join(pandeia_dir, 'roman', 'wfi', 'psfs')


def _get_default_psf_dir():
    psf_dir = get_pandeia_psf_dir()
    parent_dir = os.path.dirname(psf_dir)
    return os.path.join(parent_dir, 'default_psfs')


def reset_pandeia_psfs(originals_dir=None, suppress_output=False):
    if originals_dir is None:
        originals_dir = _get_default_psf_dir()

    psf_dir = get_pandeia_psf_dir()
    file_list = glob(originals_dir + '/wfi_imaging-f062*')

    import shutil
    for file in tqdm(file_list, disable=suppress_output):
        shutil.copy(file, psf_dir)


def update_pandeia_psfs(detector=None, detector_position=None, suppress_output=False):
    prefix = 'wfi_imaging-f062-f087-f106-f129-f146-f158_'
    wavelengths = ['0.4465e-6', '0.4725e-6', '0.5002e-6', '0.5294e-6', '0.5603e-6', '0.5931e-6', '0.6277e-6',
                   '0.6644e-6', '0.7032e-6', '0.7443e-6', '0.7878e-6', '0.8338e-6', '0.8826e-6', '0.9341e-6',
                   '0.9887e-6', '1.0465e-6', '1.1077e-6', '1.1724e-6', '1.2409e-6', '1.3134e-6', '1.3902e-6',
                   '1.4714e-6', '1.5574e-6', '1.6484e-6', '1.7447e-6', '1.8467e-6', '1.9546e-6', '2.0688e-6',
                   '2.1897e-6']

    wfi = WFI()
    wfi.filter = 'F062'
    wfi.options = _get_pandeia_psf_options()

    # set detector and detector position
    if detector is None:
        detector = get_random_detector(suppress_output)
    wfi.detector = detector
    if detector_position is None:
        detector_position = get_random_detector_pos(100, suppress_output)  # TODO refactor so input_size is meaningful
    wfi.detector_position = detector_position

    # generate monochromatic PSFs for each of the 29 wavelengths
    for wl in tqdm(wavelengths, disable=suppress_output):
        wfi.calc_psf(oversample=5,
                     fov_arcsec=4.29,
                     monochromatic=float(wl),
                     normalize='first',
                     overwrite=True,
                     outfile=os.path.join(get_pandeia_psf_dir(), f'{prefix}{wl[:-3]}.fits'))

    return detector, detector_position
