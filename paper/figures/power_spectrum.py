import hydra
import numpy as np
import os
import sys
from copy import deepcopy
from pandeia.engine.calc_utils import build_default_calc
from pyHalo.preset_models import CDM


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    array_dir = os.path.join(config.machine.array_dir, 'sample_skypy_lens')
    pickle_dir = os.path.join(config.machine.pickle_dir, 'pyhalo')

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses.test import SampleSkyPyStrongLens
    from mejiro.helpers import pyhalo, pandeia_input

    band = 'f106'
    num_samples = 100000
    grid_supersample = 3

    lens = SampleSkyPyStrongLens()

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

    np.save(os.path.join(pickle_dir, 'no_cut'), no_cut)
    np.save(os.path.join(pickle_dir, 'cut_7'), cut_7)
    np.save(os.path.join(pickle_dir, 'cut_8'), cut_8)

    no_cut_masses = [halo.mass for halo in no_cut.halos]
    cut_7_masses = [halo.mass for halo in cut_7.halos]
    cut_8_masses = [halo.mass for halo in cut_8.halos]

    np.save(os.path.join(pickle_dir, 'no_cut_masses'), no_cut_masses)
    np.save(os.path.join(pickle_dir, 'cut_7_masses'), cut_7_masses)
    np.save(os.path.join(pickle_dir, 'cut_8_masses'), cut_8_masses)

    no_cut_lens = deepcopy(lens)
    cut_7_lens = deepcopy(lens)
    cut_8_lens = deepcopy(lens)

    lenses = [no_cut_lens, cut_7_lens, cut_8_lens]

    no_cut_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(no_cut))
    cut_7_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(cut_7))
    cut_8_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(cut_8))

    no_cut_model = no_cut_lens.get_array(num_pix=51 * grid_supersample, side=5.61)
    cut_7_model = cut_7_lens.get_array(num_pix=51 * grid_supersample, side=5.61)
    cut_8_model = cut_8_lens.get_array(num_pix=51 * grid_supersample, side=5.61)

    models = [no_cut_model, cut_7_model, cut_8_model]

    titles = ['substructure_no_cut', 'substructure_cut_7', 'substructure_cut_8']

    for lens, model, title in zip(lenses, models, titles):
        calc = build_default_calc('roman', 'wfi', 'imaging')

        # set scene size settings
        calc['configuration']['max_scene_size'] = 5

        # set instrument
        calc['configuration']['instrument']['filter'] = band

        # set detector
        calc['configuration']['detector']['ma_table_name'] = 'hlwas_imaging'

        # turn off noise sources
        calc['calculation'] = pandeia_input.get_calculation_dict(init=False)

        # set background
        calc['background'] = 'none'

        # convert array from counts/sec to astronomical magnitude
        mag_array = pandeia_input._get_mag_array(lens, model, num_samples, band, suppress_output=False)

        # add point sources to Pandeia input
        norm_wave = pandeia_input._get_norm_wave(band)
        calc, _ = pandeia_input._phonion_sample(calc, mag_array, lens, num_samples, norm_wave)

        # get Pandeia image
        pandeia, _ = pandeia_input.get_pandeia_image(calc)

        np.save(os.path.join(array_dir, f'{title}.npy'), pandeia)


if __name__ == '__main__':
    main()
