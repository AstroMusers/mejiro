import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from glob import glob
import multiprocessing
import pandas as pd
from tqdm import tqdm
import time
from copy import deepcopy
from multiprocessing import Pool
import hydra

from package.helpers.test_physical_lens import TestPhysicalLens
from package.helpers.lens import Lens
from package.plots import diagnostic_plot, plot
from package.utils import util
from package.scripts import generate
from package.helpers import pyhalo


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir
    array_dir = os.path.join(array_dir, 'multiprocessing')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get path to Roman instrument params CSV
    roman_params_csv = os.path.join(repo_dir, 'data', 'roman_spacecraft_and_instrument_parameters.csv')

    # get output of SkyPy pipeline
    df = pd.read_csv(os.path.join('/data','bwedig', 'roman-population', 'data', 'dictparaggln_Area00000010.csv'))

    limit = 200
    total = deepcopy(limit)
    lens_list = []

    # generate the lens objects
    for i, row in tqdm(df.iterrows(), total=total):
        if i == limit:
            break

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
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))
        
        lens_list.append(lens)

    # split the images into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    generator = util.batch_list(lens_list, cpu_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=cpu_count) 
        output = []
        for each in pool.map(generate.main, batch):
            output.append(each)

    # unpack the output
    image_list, execution_times, num_point_sources = [], [], []
    for tuple in output:
        (image, execution_time, num_ps) = tuple
        image_list.append(image)
        execution_times.append(execution_time)
        num_point_sources.append(num_ps)

    # save images
    for i, image in enumerate(image_list):
        np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(5)}'), image)

    # save other output lists
    np.save(os.path.join(array_dir, 'skypy_output_execution_times.npy'), execution_times)
    np.save(os.path.join(array_dir, 'skypy_output_num_point_sources.npy'), num_point_sources)


if __name__ == '__main__':
    main()
