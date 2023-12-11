import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import hydra

from package.helpers import test_physical_lens, pyhalo, roman_params
from package.pandeia import pandeia_input
from package.utils import util


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir
    array_dir = os.path.join(array_dir, 'test_physical_lens')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get Roman pixel scale
    csv = os.path.join(repo_dir, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    roman_pixel_scale = roman_params.RomanParameters(csv).get_pixel_scale()
    
    num_pix = 51  # (45 + (2 * 3))
    side = 5.61  # (4.95 + (2 * 0.33))
    grid_oversample = 5
    num_samples_list = [1000, 5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000, 7500000, 10000000]
    num_samples_list = [int(i) for i in num_samples_list]  # convert to list of int as scientific notation in Python gives float

    execution_times = []

    # use test lens
    lens = test_physical_lens.TestPhysicalLens()

    # add CDM subhalos; NB same subhalo population for all
    lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))

    for num_samples in tqdm(num_samples_list):
        model = lens.get_array(num_pix=num_pix * grid_oversample, side=side)

        # build Pandeia input
        calc, _ = pandeia_input.build_pandeia_calc(csv=csv,
                                                array=model, 
                                                lens=lens, 
                                                band='f106',
                                                num_samples=num_samples,
                                                suppress_output=True)

        # do Pandeia calculation        
        image, execution_time = pandeia_input.get_pandeia_image(calc,
                                                                suppress_output=True)
        execution_times.append(execution_time)
        
        # save detector image
        np.save(os.path.join(array_dir, f'test_physical_lens_image_{num_samples}'), image)

    # save list of execution times
    np.save(os.path.join(array_dir, 'test_physical_lens_execution_times.npy'), execution_times)


if __name__ == '__main__':
    main()
