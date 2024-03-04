import os
import sys

import hydra
import numpy as np
from pandeia.engine.calc_utils import build_default_calc


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(config):
    array_dir = os.path.join(config.machine.array_dir, 'sample_skypy_lens')
    pickle_dir = os.path.join(config.machine.pickle_dir, 'pyhalo')

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.helpers import pyhalo, pandeia_input

    band = 'F129'
    num_samples = 100000
    grid_supersample = 3

    lens = SampleStrongLens()

    # add CDM subhalos; NB same subhalo population for all
    lens.add_subhalos(*pyhalo.unpickle_subhalos(os.path.join(pickle_dir, 'cdm_subhalos_tuple')))

    array = lens.get_array(num_pix=51 * grid_supersample, side=5.61, band=band)

    calc = build_default_calc('roman', 'wfi', 'imaging')

    # set scene size settings
    calc['configuration']['max_scene_size'] = 5

    # set instrument
    calc['configuration']['instrument']['filter'] = band.lower()

    # set detector
    calc['configuration']['detector']['ma_table_name'] = 'hlwas_imaging'

    # turn off noise sources
    calc['calculation'] = pandeia_input.get_calculation_dict(init=True)  # TODO toggle back

    # set background
    calc['background'] = 'none'

    # convert array from amp to counts/sec
    cps_array = pandeia_input._get_cps_array(lens, array, num_samples, band, background=None)

    # convert array from counts/sec to astronomical magnitude
    mag_array = pandeia_input._convert_cps_to_magnitude(cps_array, band)

    # add point sources to Pandeia input
    norm_wave = pandeia_input._get_norm_wave(band)
    calc, _ = pandeia_input._phonion_sample(calc, mag_array, lens, norm_wave, suppress_output=False)

    # get Pandeia image
    pandeia, _ = pandeia_input.get_pandeia_image(calc)

    np.save(os.path.join(array_dir, f'no_background_{band}_{grid_supersample}_{num_samples}.npy'), pandeia)  # noise_or_


if __name__ == '__main__':
    main()
