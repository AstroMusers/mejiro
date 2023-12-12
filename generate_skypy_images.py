import datetime
import os
import sys
import time

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

from package.helpers import pyhalo, roman_params, pandeia_input
from package.lenses.lens import Lens
from package.utils import util


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir
    array_dir = os.path.join(array_dir, 'skypy')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get Roman pixel scale
    csv = os.path.join(repo_dir, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    roman_pixel_scale = roman_params.RomanParameters(csv).get_pixel_scale()

    # read in SkyPy pipeline output
    csv_path = os.path.join(repo_dir, 'data', 'dictparaggln_Area00000010.csv')
    df = pd.read_csv(csv_path)

    # set number of images to select
    limit = 16
    total = limit
    lens_list = []
    lens_execution_times, pandeia_execution_times, estimated_times = [], [], []

    print('Building lenses from SkyPy pipeline output')
    for i, row in tqdm(df.iterrows(), total=total):
        start = time.time()

        if i == limit:
            break

        # # select only the cool ones lmao
        # if row['numbimag'] == 1.0:
        #     # print('Skipping uncool lens')
        #     limit += 1
        #     continue

        lens = Lens(z_lens=row['redslens'],
                    z_source=row['redssour'],
                    sigma_v=row['velodisp'],
                    lens_x=row['xposlens'],
                    lens_y=row['yposlens'],
                    source_x=row['xpossour'],
                    source_y=row['ypossour'],
                    mag_lens=row['magtlensF106'],
                    mag_source=row['magtsourF106'])

        # add CDM subhalos
        try:
            lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))
        except:
            # print(f'\nSkipping {lens.z_lens}, {lens.z_source}')
            limit += 1
            continue

        lens_list.append(lens)

        stop = time.time()
        execution_time = str(datetime.timedelta(seconds=round(stop - start)))
        lens_execution_times.append(execution_time)

    print('Generating Pandeia images')
    for i, lens in tqdm(enumerate(lens_list)):
        grid_oversample = 1
        num_samples = 1000

        model = lens.get_array(num_pix=90 * grid_oversample, side=10.)

        # build Pandeia input
        calc, num_point_sources = pandeia_input.build_pandeia_calc(csv=csv,
                                                                   array=model,
                                                                   lens=lens,
                                                                   side=10.,
                                                                   band='f106',
                                                                   num_samples=num_samples)

        # get estimated calculation time
        estimated_times.append(pandeia_input.estimate_calculation_time(num_point_sources))

        # do Pandeia calculation        
        image, execution_time = pandeia_input.get_pandeia_image(calc)
        pandeia_execution_times.append(execution_time)

        # save detector image
        np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(5)}.npy'), image)

    # save lists of execution times
    np.save(os.path.join(array_dir, 'skypy_output_lens_execution_times.npy'), lens_execution_times)
    np.save(os.path.join(array_dir, 'skypy_output_pandeia_execution_times.npy'), pandeia_execution_times)
    np.save(os.path.join(array_dir, 'skypy_output_estimated_times.npy'), estimated_times)


if __name__ == '__main__':
    main()
