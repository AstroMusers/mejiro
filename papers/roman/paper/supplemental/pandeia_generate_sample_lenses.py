import hydra
import numpy as np
import os
import sys
import time
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    array_dir, pickle_dir, repo_dir = config.machine.array_dir, config.machine.pickle_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.helpers import pandeia_input
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.utils import util

    output_dir = os.path.join(array_dir, 'sample_skypy_lens')
    # util.create_directory_if_not_exists(output_dir)
    # util.clear_directory(output_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    band = 'F106'
    num_pix = 51
    side = 5.61
    output_size = 45
    # grid_oversample_list = [1, 3, 5]  # 
    grid_oversample_list = [1]
    # num_samples_list = [100]  # , 1000, 10000, 100000, 1000000, 10000000
    num_samples_list = [int(1e10), int(1e11), int(1e12), int(1e13), int(1e14)]

    # use test lens
    lens = SampleStrongLens()

    # add CDM subhalos; NB same subhalo population for all
    realization = util.unpickle(os.path.join(pickle_dir, 'cdm_subhalos_for_sample.pkl'))
    lens.add_subhalos(realization)

    # # generate sky background and reshape for each grid oversampling
    # bkgs = []
    # background = bkg.get_high_galactic_lat_bkg((num_pix, num_pix), band, seed=seed)
    # for grid_oversample in grid_oversample_list:
    #     reshaped_bkg = util.resize_with_pixels_centered(background, grid_oversample)
    #     np.save(os.path.join(array_dir, f'bkg_{grid_oversample}'), reshaped_bkg)
    #     bkgs.append(reshaped_bkg)

    # generate each image
    for i, grid_oversample in enumerate(grid_oversample_list):
        execution_time = []
        execution_time_x = []

        for num_samples in tqdm(num_samples_list):
            start = time.time()

            model = lens.get_array(num_pix=num_pix * grid_oversample,
                                   side=side, band=band)

            # build Pandeia input
            calc, _ = pandeia_input.build_pandeia_calc(model, lens, background=None, band=band,
                                                       max_scene_size=output_size, num_samples=num_samples,
                                                       suppress_output=False)

            # do Pandeia calculation        
            image, _ = pandeia_input.get_pandeia_image(calc, suppress_output=False)

            # center crop
            image = util.center_crop_image(image, (output_size, output_size))

            # save image
            np.save(os.path.join(output_dir, f'sample_skypy_lens_{grid_oversample}_{num_samples}'), image)

            stop = time.time()
            execution_time.append(stop - start)
            execution_time_x.append((grid_oversample, num_samples))

        np.save(os.path.join(output_dir, f'execution_time_{grid_oversample}_last'), execution_time)
        np.save(os.path.join(output_dir, f'execution_time_x_{grid_oversample}_last'), execution_time_x)


if __name__ == '__main__':
    main()
