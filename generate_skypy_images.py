import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from package.helpers.lens import Lens
from package.pandeia import pandeia_input
from package.pandeia.pandeia_output import PandeiaOutput


def main():
    cwd = os.getcwd()
    while os.path.basename(os.path.normpath(cwd)) != 'roman-pandeia':
        cwd = os.path.dirname(cwd)
    repo_path = cwd
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    csv = os.path.join(repo_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    array_dir = os.path.join(repo_path, 'output', 'arrays')
    figure_dir = os.path.join(repo_path, 'figures')
    data_dir = os.path.join('/data','bwedig', 'roman-population', 'data')
    csv_path = os.path.join(data_dir, 'dictparaggln_Area00000010.csv')
    df = pd.read_csv(csv_path)

    # set number of images to select
    limit = 16
    lens_list = []
    execution_times = []

    # build list of Lenses from SkyPy output
    for i, row in df.iterrows():
        if i == limit:
            break
        lens = Lens(z_lens = row['redssour'], 
                    z_source=row['redslens'], 
                    sigma_v=row['velodisp'], 
                    lens_x=row['xposlens'], 
                    lens_y=row['yposlens'], 
                    source_x=row['xpossour'], 
                    source_y=row['ypossour'], 
                    mag_lens=row['magtlensF087'], 
                    mag_source=row['magtsourF087'])
        lens_list.append(lens)

    # generate Pandeia images
    for i, lens in enumerate(lens_list):
        model = lens.get_array(num_pix=90, side=10.)

        # build Pandeia input
        calc = pandeia_input.build_pandeia_calc(csv=csv,
                                                array=model, 
                                                lens=lens, 
                                                band='f106', 
                                                oversample_factor=1)

        # do Pandeia calculation        
        results, execution_time = pandeia_input.get_pandeia_results(calc)

        pandeia_output = PandeiaOutput(results)
        execution_times.append(execution_time)
        
        # save detector image
        np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(5)}'), pandeia_output.get_image())

    # save list of execution times
    np.save(os.path.join(array_dir, 'skypy_output_execution_times.npy'), execution_times)


if __name__ == '__main__':
    main()
