import os
from glob import glob

import astropy.io.fits as pyfits
import galsim
from galsim import roman
from tqdm import tqdm
from webbpsf.roman import WFI

from mejiro.helpers import gs
from mejiro.utils import util


def get_webbpsf_psf(band, detector, detector_position, oversample, check_cache=False, psf_cache_dir=None,
                    suppress_output=True):
    # first, check if it exists in the cache
    if check_cache:
        if psf_cache_dir is None:
            import mejiro
            module_path = os.path.dirname(mejiro.__file__)
            psf_cache_dir = os.path.join(module_path, 'data', 'cached_psfs')
        psf_path = glob(os.path.join(psf_cache_dir,
                                     f'{band}_{detector}_{detector_position[0]}_{detector_position[1]}_{oversample}.pkl'))
        if len(psf_path) == 1:
            if not suppress_output: print(f'Loading cached PSF: {psf_path[0]}')
            return util.unpickle(psf_path[0])
        else:
            if not suppress_output: print(
                f'PSF {band} SCA{str(detector).zfill(2)} {detector_position} {oversample} not found in cache {psf_cache_dir}')

    # set PSF parameters
    wfi = WFI()
    wfi.filter = band.upper()
    wfi.detector = detector_int_to_sca(detector)
    wfi.detector_position = detector_position

    # generate PSF in WebbPSF
    psf = wfi.calc_psf(oversample=oversample)

    return kernel_to_galsim_interpolated_image(psf[0].data, oversample, pixel_scale=wfi.pixelscale)


def kernel_to_galsim_interpolated_image(kernel, oversample, pixel_scale=0.11):
    oversampled_pixel_scale = pixel_scale / oversample
    psf_image = galsim.Image(kernel, scale=oversampled_pixel_scale)
    return galsim.InterpolatedImage(psf_image)


# def get_gaussian_psf(fwhm, oversample, pixel_scale=0.11):
#     from lenstronomy.Data.psf import PSF

#     kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': pixel_scale, 'truncation': 6}
#     psf_class = PSF(**kwargs_psf)

#     # import PSF to GalSim
#     oversampled_pixel_scale = pixel_scale / oversample
#     psf_image = galsim.Image(psf_class.kernel_pixel, scale=oversampled_pixel_scale)
#     return galsim.InterpolatedImage(psf_image)


def get_gaussian_psf(fwhm):
    return galsim.Gaussian(fwhm=fwhm, flux=1.)


def get_galsim_psf(band, detector, detector_position, pupil_bin=1):
    return roman.getPSF(SCA=detector,
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


def get_psf_id_string(band, detector, detector_position, oversample):
    return f'{band}_{detector}_{detector_position[0]}_{detector_position[1]}_{oversample}'


def load_cached_psf(psf_id_string, psf_cache_dir=None, suppress_output=True):
    if psf_cache_dir is None:
        import mejiro
        module_path = os.path.dirname(mejiro.__file__)
        psf_cache_dir = os.path.join(module_path, 'data', 'cached_psfs')
    psf_path = os.path.join(psf_cache_dir, f'{psf_id_string}.pkl')
    if os.path.exists(psf_path):
        if not suppress_output: print(f'Loading cached PSF: {psf_path}')
        return util.unpickle(psf_path)
    else:
        if not suppress_output: print(f'PSF {psf_id_string} not found in cache {psf_cache_dir}')
        band, detector, detector_pos_x, detector_pos_y, oversample = psf_id_string.split('_')
        psf_kernel = get_psf_kernel(band, int(detector), (int(detector_pos_x), int(detector_pos_y)), int(oversample))
        util.pickle(psf_path, psf_kernel)
        if not suppress_output: print(f'Generated and cached PSF: {psf_path}')
        return kernel_to_galsim_interpolated_image(psf_kernel, int(oversample))


def detector_int_to_sca(detector):
    # detector might be int or string, and WebbPSF expects 'SCA01', 'SCA02', etc.
    if type(detector) == int:
        return f'SCA{str(detector).zfill(2)}'
    # TODO handle a few more cases


def get_psf_kernel(band, detector, detector_position, oversample=5, fov_arcsec=None, save=None):
    wfi = WFI()
    wfi.filter = band.upper()
    wfi.detector = detector_int_to_sca(detector)
    wfi.detector_position = detector_position
    psf = wfi.calc_psf(oversample=oversample, fov_arcsec=fov_arcsec)
    if save is not None:
        psf.writeto(save, overwrite=True)
    return psf[0].data


# TODO collapse these methods into one that randomizes if None is provided

def get_random_psf_kernel(band, oversample=5, save=None, suppress_output=False):
    wfi = WFI()
    wfi.filter = band.upper()
    wfi.detector = detector_int_to_sca(gs.get_random_detector(suppress_output))
    wfi.detector_position = gs.get_random_detector_pos(100,
                                                       oversample,
                                                       suppress_output)  # TODO refactor so input_size is meaningful
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


def get_pandeia_psf_dir():
    pandeia_dir = os.environ['pandeia_refdata']
    return os.path.join(pandeia_dir, 'roman', 'wfi', 'psfs')


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
        detector = detector_int_to_sca(gs.get_random_detector(suppress_output))
    wfi.detector = detector_int_to_sca(detector)
    if detector_position is None:
        detector_position = gs.get_random_detector_pos(100,
                                                       oversample,
                                                       suppress_output)  # TODO refactor so input_size is meaningful
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


def _get_pandeia_psf_options():
    return {'source_offset_r': 0.,
            'source_offset_theta': 0.,
            'pupil_shift_x': 0.,
            'pupil_shift_y': 0.,
            'output_mode': 'oversampled',
            'jitter': 'gaussian',
            'jitter_sigma': 0.012}


def _get_default_psf_dir():
    psf_dir = get_pandeia_psf_dir()
    parent_dir = os.path.dirname(psf_dir)
    return os.path.join(parent_dir, 'default_psfs')
