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

    # TODO collect num=1000 lenses

    # TODO make a copy then add a given subhalo population

    # TODO make another copy, etc.

    lens = SampleStrongLens()

    # generate subhalos
    no_cut = CDM(lens.z_lens,
                 lens.z_source,
                 cone_opening_angle_arcsec=6.,
                 LOS_normalization=0.,
                 log_mlow=6.,
                 log_mhigh=10.)

    cut_7 = CDM(lens.z_lens,
                lens.z_source,
                cone_opening_angle_arcsec=6.,
                LOS_normalization=0.,
                log_mlow=7.,
                log_mhigh=10.)

    cut_8 = CDM(lens.z_lens,
                lens.z_source,
                cone_opening_angle_arcsec=6.,
                LOS_normalization=0.,
                log_mlow=8.,
                log_mhigh=10.)

    util.pickle(os.path.join(save_dir, 'no_cut'), no_cut)
    util.pickle(os.path.join(save_dir, 'cut_7'), cut_7)
    util.pickle(os.path.join(save_dir, 'cut_8'), cut_8)

    no_cut_masses = [halo.mass for halo in no_cut.halos]
    cut_7_masses = [halo.mass for halo in cut_7.halos]
    cut_8_masses = [halo.mass for halo in cut_8.halos]

    np.save(os.path.join(save_dir, 'no_cut_masses'), no_cut_masses)
    np.save(os.path.join(save_dir, 'cut_7_masses'), cut_7_masses)
    np.save(os.path.join(save_dir, 'cut_8_masses'), cut_8_masses)

    no_cut_lens = deepcopy(lens)
    cut_7_lens = deepcopy(lens)
    cut_8_lens = deepcopy(lens)

    no_cut_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(no_cut))
    cut_7_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(cut_7))
    cut_8_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(cut_8))

    no_cut_model = no_cut_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
    cut_7_model = cut_7_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
    cut_8_model = cut_8_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)

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
