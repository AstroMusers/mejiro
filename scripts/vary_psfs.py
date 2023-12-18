import os
import sys
from copy import deepcopy

import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(config):
    array_dir, pickle_dir, repo_dir = config.machine.array_dir, config.machine.pickle_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.helpers import pyhalo, pandeia_input
    from mejiro.lenses.test import SampleSkyPyLens
    from mejiro.utils import util

    array_dir = os.path.join(array_dir, 'psf_compare')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # set parameters
    num_samples = 100000
    grid_oversample = 5
    num_pix = 45
    side = 4.95
    band = 'f106'

    # use test lens
    lens = SampleSkyPyLens()

    # add CDM subhalos; NB same subhalo population for all
    pickle_dir = os.path.join(pickle_dir, 'pyhalo')
    lens.add_subhalos(*pyhalo.unpickle_subhalos(os.path.join(pickle_dir, 'cdm_subhalos_tuple')))

    # no PSF
    no_psf_lens = deepcopy(lens)
    kwargs_psf_none = {'psf_type': 'NONE'}
    no_psf = no_psf_lens.get_array(num_pix=num_pix * grid_oversample, kwargs_psf=kwargs_psf_none, side=side)
    np.save(os.path.join(array_dir, f'no_psf_{grid_oversample}_{num_samples}.npy'), no_psf)
    assert no_psf.shape == (45, 45)

    # Gaussian PSF
    psf_fwhm = {
        'f062': 0.058,
        'f087': 0.073,
        'f106': 0.087,
        'f129': 0.105,
        'f158': 0.127,
        'f184': 0.151,
        'f213': 0.175,
        'f146': 0.105
    }
    gaussian_psf_lens = deepcopy(lens)
    kwargs_psf_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm.get(band)}
    gaussian_psf = gaussian_psf_lens.get_array(num_pix=num_pix * grid_oversample, kwargs_psf=kwargs_psf_gaussian, side=side)
    np.save(os.path.join(array_dir, f'gaussian_psf_{grid_oversample}_{num_samples}.npy'), gaussian_psf)

    # Pandeia PSF, no noise or background
    pandeia_lens = deepcopy(lens)
    pandeia_model = pandeia_lens.get_array(num_pix=num_pix * grid_oversample, side=side)
    calc, _ = pandeia_input.build_pandeia_calc(pandeia_model, pandeia_lens, num_samples=num_samples, suppress_output=True) 
    image, _ = pandeia_input.get_pandeia_image(calc, suppress_output=True)
    assert image.shape == (45, 45)
    np.save(os.path.join(array_dir, f'pandeia_{grid_oversample}_{num_samples}'), image)


if __name__ == '__main__':
    main()
