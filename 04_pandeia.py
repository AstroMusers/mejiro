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
import pickle

from package.helpers.test_physical_lens import TestPhysicalLens
from package.helpers.lens import Lens
from package.plots import diagnostic_plot, plot
from package.utils import util
from package.helpers import pyhalo
from package.pandeia import pandeia_input


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir
    array_dir = os.path.join(array_dir, 'skypy_output')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get lenstronomy models
    npy_list = glob(array_dir + '/skypy_output0*.npy')
    array_list = [np.load(i) for i in npy_list]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    generator = util.batch_list(array_list, process_count)
    batches = list(generator)

    # process the batches
    i = 0
    for batch in tqdm(batches):
        pool = Pool(processes=process_count) 
        for output in pool.map(get_image, batch):
            (image, execution_time) = output
            np.save(os.path.join(array_dir, f'skypy_output_pandeia_{str(i).zfill(8)}.npy'), image)
            i += 1


# TODO fix - this might require quite a bit of refactoring of pandeia_input.py
def get_image(array):
    calc, num_point_sources = pandeia_input.build_pandeia_calc(csv, array, lens, band, side=side, num_samples=num_samples, suppress_output=True)

    return pandeia_input.get_pandeia_image(calc, suppress_output=False)


if __name__ == '__main__':
    main()
