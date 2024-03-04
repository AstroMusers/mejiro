import os
import sys
import time

import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.helpers import pandeia_input, bkg, psf
    from mejiro.utils import util

    # directory to read from
    input_dir = config.machine.dir_03

    # directory to write the output to
    output_dir = os.path.join(config.machine.pipeline_dir, '04_off_axis')  # config.machine.dir_04
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # open pickled lens list
    # TODO LIMIT IS TEMP
    limit = 4
    lens_list = util.unpickle_all(input_dir, 'lens_', limit)

    execution_times = []

    # unpack pipeline_params
    pipeline_params = util.hydra_to_dict(config.pipeline)
    bands = pipeline_params['band']
    grid_oversample = pipeline_params['grid_oversample']
    max_scene_size = pipeline_params['max_scene_size']
    num_samples = pipeline_params['num_samples']

    for lens in tqdm(lens_list):
        # update PSFs
        # TODO get random detector and detector position
        psf.update_pandeia_psfs()

        # load an array to get its shape
        num_pix, _ = np.load(f'{input_dir}/array_{lens.uid}_{bands[0]}.npy').shape

        # generate sky background
        print('Generating sky background...')
        bkgs = bkg.get_high_galactic_lat_bkg((num_pix, num_pix), bands, seed=None)
        reshaped_bkgs = [util.resize_with_pixels_centered(i, grid_oversample) for i in bkgs]

        execution_times = []
        for i, band in enumerate(bands):
            print(f'Generating image for lens {lens.uid}, band {band}...')

            # load the appropriate array
            array = np.load(f'{input_dir}/array_{lens.uid}_{band}.npy')

            # build Pandeia input
            calc, _ = pandeia_input.build_pandeia_calc(array, lens, background=reshaped_bkgs[i], noise=True, band=band,
                                                       max_scene_size=max_scene_size,
                                                       num_samples=num_samples, suppress_output=False)

            # generate Pandeia image and save
            image, execution_time = pandeia_input.get_pandeia_image(calc, suppress_output=False)
            np.save(os.path.join(output_dir, f'pandeia_{lens.uid}_{band}.npy'), image)
            print(f'Finished lens {lens.uid}, band {band}')

            execution_times.append(execution_time)

    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
