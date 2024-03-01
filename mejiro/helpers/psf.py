import astropy.io.fits as pyfits
import os
import random
from glob import glob
from tqdm import tqdm
from webbpsf import roman


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
    wavelengths = ['0.4465e-6','0.4725e-6', '0.5002e-6', '0.5294e-6', '0.5603e-6', '0.5931e-6', '0.6277e-6', '0.6644e-6', '0.7032e-6', '0.7443e-6', '0.7878e-6', '0.8338e-6', '0.8826e-6', '0.9341e-6', '0.9887e-6', '1.0465e-6', '1.1077e-6', '1.1724e-6', '1.2409e-6', '1.3134e-6', '1.3902e-6', '1.4714e-6', '1.5574e-6', '1.6484e-6', '1.7447e-6', '1.8467e-6', '1.9546e-6', '2.0688e-6', '2.1897e-6']

    wfi = roman.WFI()
    wfi.filter = 'F062'
    wfi.options = _get_pandeia_psf_options()

    # set detector and detector position
    if detector is None:
        detector = get_random_detector(suppress_output)
    wfi.detector = detector
    if detector_position is None:
        detector_position = get_random_position(suppress_output)
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


def get_kwargs_psf(kernel, oversample):
    return {
        'psf_type': 'PIXEL',
        'kernel_point_source': kernel,
        'point_source_supersampling_factor': oversample
    }


def get_kernel_from_calc_psf(calc_psf):
    return calc_psf['DET_SAMP'].data


# TODO refactor this to accept a detector_position tuple instead of x, y
def get_psf_kernel(band, detector, detector_position, oversample=5, save=None, suppress_output=False):
    wfi = get_instrument(band)
    wfi.detector = detector
    wfi.detector_position = detector_position
    psf = wfi.calc_psf(oversample=oversample)
    if save is not None:
        psf.writeto(save, overwrite=True)
    return get_kernel_from_calc_psf(psf)


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


# TODO use tuple instead of x, y
def get_random_position(suppress_output=False):
    # Roman WFI detectors are 4096x4096 pixels, but the outermost four rows and columns are reference pixels
    x, y = random.randrange(4, 4092), random.randrange(4, 4092)
    if not suppress_output:
        print(f'Detector position: {x}, {y}')
    return (x, y)


def get_random_detector(suppress_output=False):
    detector = random.choice(roman.WFI().detector_list)
    if not suppress_output:
        print(f'Detector: {detector}')
    return detector
