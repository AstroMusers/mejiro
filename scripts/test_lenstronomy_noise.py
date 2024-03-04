import os
import sys
import time
from copy import deepcopy

import hydra
import numpy as np
from pandeia.engine.calc_utils import build_default_calc
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(config):
    array_dir, pickle_dir, repo_dir = config.machine.array_dir, config.machine.pickle_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.helpers import pyhalo, pandeia_input, lenstronomy_sim
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.utils import util

    array_dir = os.path.join(array_dir, 'sample_skypy_lens', 'lenstronomy_noise')
    util.create_directory_if_not_exists(array_dir)
    util.clear_directory(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    grid_oversample_list = [1, 3, 5]
    num_samples_list = [100, 1000, 10000, 100000, 1000000, 10000000]

    band = 'f106'

    # use test lens
    lens = SampleStrongLens()

    # add CDM subhalos; NB same subhalo population for all
    pickle_dir = os.path.join(pickle_dir, 'pyhalo')
    lens.add_subhalos(*pyhalo.unpickle_subhalos(os.path.join(pickle_dir, 'cdm_subhalos_tuple')))

    for grid_oversample in grid_oversample_list:
        execution_time = []
        execution_time_x = []

        grid_lens = deepcopy(lens)

        array = grid_lens.get_array(num_pix=51 * grid_oversample,
                                    side=5.61)  # .get_array(num_pix=97 * grid_oversample, side=10.67)

        # generate noise and save
        noise = lenstronomy_sim.get_background_noise(grid_lens, array, band)
        np.save(os.path.join(array_dir, f'noise_{grid_oversample}'), noise)

        # add noise and save
        array += noise
        np.save(os.path.join(array_dir, f'input_{grid_oversample}'), array)

        for num_samples in tqdm(num_samples_list):
            start = time.time()

            # build Pandeia input
            calc = build_default_calc('roman', 'wfi', 'imaging')

            # set scene size settings
            calc['configuration']['max_scene_size'] = 5.

            # set instrument
            calc['configuration']['instrument']['filter'] = band.lower()  # e.g. 'f106'

            # set detector
            calc['configuration']['detector']['ma_table_name'] = 'hlwas_imaging'

            # turn on noise sources
            calc['calculation'] = pandeia_input.get_calculation_dict(init=True)

            # turn off Pandeia background
            calc['background'] = 'none'

            # array += get_background_noise(lens, band)

            # convert array from counts/sec to astronomical magnitude
            mag_array = pandeia_input._get_mag_array(grid_lens, array, num_samples, band, suppress_output=False)

            # add point sources to Pandeia input
            norm_wave = pandeia_input._get_norm_wave(band)
            calc, num_point_sources = pandeia_input._phonion_sample(calc, mag_array, grid_lens, num_samples, norm_wave,
                                                                    suppress_output=False)

            print(f'Estimated calculation time: {pandeia_input.estimate_calculation_time(num_point_sources)}')

            # do Pandeia calculation        
            image, _ = pandeia_input.get_pandeia_image(calc, suppress_output=False)
            assert image.shape == (45, 45)  # 91, 91

            # save image
            np.save(os.path.join(array_dir, f'sample_skypy_lens_lenstronomy_noise_{grid_oversample}_{num_samples}'),
                    image)

            stop = time.time()
            util.print_execution_time(start, stop)
            execution_time.append(stop - start)
            execution_time_x.append((grid_oversample, num_samples))

        np.save(os.path.join(array_dir, f'execution_time_{grid_oversample}'), execution_time)
        np.save(os.path.join(array_dir, f'execution_time_x_{grid_oversample}'), execution_time_x)


if __name__ == '__main__':
    main()
