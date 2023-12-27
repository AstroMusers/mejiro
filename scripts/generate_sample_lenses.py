import os
import sys
import time

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

    array_dir = os.path.join(array_dir, 'sample_skypy_lens')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    grid_oversample_list = [1, 3, 5]
    num_samples_list = [100, 1000, 10000, 100000, 1000000, 10000000]

    # use test lens
    lens = SampleSkyPyLens()

    # add CDM subhalos; NB same subhalo population for all
    pickle_dir = os.path.join(pickle_dir, 'pyhalo')
    lens.add_subhalos(*pyhalo.unpickle_subhalos(os.path.join(pickle_dir, 'cdm_subhalos_tuple')))

    for grid_oversample in grid_oversample_list:
        execution_time = []
        execution_time_x = []

        for num_samples in tqdm(num_samples_list):
            start = time.time()

            model = lens.get_array(num_pix=97 * grid_oversample, side=10.67)  # .get_array(num_pix=51 * grid_oversample, side=5.61)

            # build Pandeia input
            calc, _ = pandeia_input.build_pandeia_calc(model, lens, max_scene_size=10., num_samples=num_samples, suppress_output=True)

            # do Pandeia calculation        
            image, _ = pandeia_input.get_pandeia_image(calc, suppress_output=True)
            assert image.shape == (91, 91) # 45, 45

            # save image
            np.save(os.path.join(array_dir, f'sample_skypy_lens_{grid_oversample}_{num_samples}'), image)

            stop = time.time()
            execution_time.append(stop - start)
            execution_time_x.append((grid_oversample, num_samples))

        np.save(os.path.join(array_dir, f'execution_time_{grid_oversample}'), execution_time)
        np.save(os.path.join(array_dir, f'execution_time_x_{grid_oversample}'), execution_time_x)


if __name__ == '__main__':
    main()
