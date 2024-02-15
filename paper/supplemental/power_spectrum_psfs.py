import hydra
import numpy as np
import os
import sys
from copy import deepcopy
from pyHalo.preset_models import CDM


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.helpers import gs, pyhalo
    from mejiro.utils import util

    # set save path for everything
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectrum_galsim')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    band = 'F184'
    grid_oversample = 3
    num_pix = 45
    side = 4.95

    # use test lens
    lens = SampleStrongLens()

    # add CDM subhalos; NB same subhalo population for all
    pickle_dir = os.path.join(pickle_dir, 'pyhalo')
    lens.add_subhalos(*pyhalo.unpickle_subhalos(os.path.join(pickle_dir, 'cdm_subhalos_tuple')))

    
    no_psf_lens = deepcopy(lens)
    gaussian_psf_lens = deepcopy(lens)
    webbpsf_lens = deepcopy(lens)

    no_psf = no_psf_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band, kwargs_psf={'psf_type': 'NONE'})
    gaussian_psf = gaussian_psf_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
    webbpsf = webbpsf_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)

    np.save(os.path.join(save_dir, 'no_cut_model'), no_cut_model)
    np.save(os.path.join(save_dir, 'cut_7_model'), cut_7_model)
    np.save(os.path.join(save_dir, 'cut_8_model'), cut_8_model)

    lenses = [no_cut_lens, cut_7_lens, cut_8_lens]
    models = [no_cut_model, cut_7_model, cut_8_model]
    titles = ['substructure_no_cut', 'substructure_cut_7', 'substructure_cut_8']

    for lens, model, title in zip(lenses, models, titles):
        gs_images, _ = gs.get_images(lens, model, band, input_size=num_pix, output_size=num_pix, grid_oversample=grid_oversample)
        np.save(os.path.join(save_dir, f'{title}.npy'), gs_images[0])


if __name__ == '__main__':
    main()
