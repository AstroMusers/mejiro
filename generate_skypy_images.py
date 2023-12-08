import os
import sys
import time
import datetime
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from package.helpers.lens import Lens
from package.helpers import pyhalo
from package.pandeia import pandeia_input
from package.pandeia.pandeia_output import PandeiaOutput

np.random.seed(92)


def main():
    cwd = os.getcwd()
    while os.path.basename(os.path.normpath(cwd)) != 'roman-pandeia':
        cwd = os.path.dirname(cwd)
    repo_path = cwd
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    csv = os.path.join(repo_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    array_dir = os.path.join(repo_path, 'output', 'arrays', 'skypy_test')
    figure_dir = os.path.join(repo_path, 'figures')
    data_dir = os.path.join('/data','bwedig', 'roman-population', 'data')
    csv_path = os.path.join(data_dir, 'dictparaggln_Area00000010.csv')
    df = pd.read_csv(csv_path)

    # set number of images to select
    limit = 16
    lens_list = []
    lens_execution_times, pandeia_execution_times = [], []

    # build list of Lenses from SkyPy output
    print('Building lenses from SkyPy pipeline output')
    for i, row in tqdm(df.iterrows(), total=limit):
        # print(f'New loop: i={i}, limit={limit}')
        start = time.time()

        if i == limit:
            break

        # select only the cool ones lmao
        if row['numbimag'] == 1.0:
            # print('Skipping uncool lens')
            limit += 1
            continue

        lens = Lens(z_lens = row['redslens'], 
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

    # generate Pandeia images
    print('Generating Pandeia images')
    for i, lens in tqdm(enumerate(lens_list)):
        grid_oversample = 3
        num_samples = 10000

        model = lens.get_array(num_pix=90 * grid_oversample, side=10.)

        # build Pandeia input
        calc = pandeia_input.build_pandeia_calc(csv=csv,
                                                array=model, 
                                                lens=lens, 
                                                band='f106', 
                                                num_samples=num_samples)

        # do Pandeia calculation        
        results, execution_time = pandeia_input.get_pandeia_results(calc)

        pandeia_output = PandeiaOutput(results)
        pandeia_execution_times.append(execution_time)
        
        # save detector image
        np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(5)}'), pandeia_output.get_image())

    # save lists of execution times
    np.save(os.path.join(array_dir, 'skypy_output_lens_execution_times.npy'), lens_execution_times)
    np.save(os.path.join(array_dir, 'skypy_output_pandeia_execution_times.npy'), pandeia_execution_times)


if __name__ == '__main__':
    main()
