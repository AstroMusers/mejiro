import os
import sys
from copy import deepcopy

import hydra
import numpy as np


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util

    # set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'output', 'psf_compare')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    # set imaging params
    band = 'F184'
    grid_oversample = 5
    num_pix = 45
    side = 4.95


def generate_power_spectra(tuple):
    from mejiro.analysis import ft
    from mejiro.helpers import psf

    # generate PSF power spectra
    # no PSF
    no_psf_lens = deepcopy(lens)
    no_psf = no_psf_lens.get_array(num_pix=num_pix, side=side, band=band)

    # Gaussian PSF
    gaussian_psf_lens = deepcopy(lens)
    # PSF FWHM for F184; see https://roman.gsfc.nasa.gov/science/WFI_technical.html
    kwargs_psf_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': 0.151}  # TODO get FWHM from roman_params
    gaussian_psf = gaussian_psf_lens.get_array(num_pix=num_pix, side=side, band=band, kwargs_psf=kwargs_psf_gaussian)

    # WebbPSF
    webbpsf_lens = deepcopy(lens)
    webbpsf_kernel = psf.get_psf_kernel(band, detector=1, detector_position=(2048, 2048), oversample=grid_oversample)
    kwargs_webbpsf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': webbpsf_kernel,
        'point_source_supersampling_factor': 5
    }
    webbpsf_psf = webbpsf_lens.get_array(band=band, num_pix=num_pix, kwargs_psf=kwargs_webbpsf, side=side)

    no_psf_power = ft.power_spectrum(no_psf)
    gaussian_psf_power = ft.power_spectrum(gaussian_psf)
    webbpsf_psf_power = ft.power_spectrum(webbpsf_psf)

    np.save(os.path.join(lens_dir, f'power_spectrum_{title}_psf_none.npy'), no_psf_power)
    np.save(os.path.join(lens_dir, f'power_spectrum_{title}_psf_gaussian.npy'), gaussian_psf_power)
    np.save(os.path.join(lens_dir, f'power_spectrum_{title}_psf_webbpsf.npy'), webbpsf_psf_power)


if __name__ == '__main__':
    main()
