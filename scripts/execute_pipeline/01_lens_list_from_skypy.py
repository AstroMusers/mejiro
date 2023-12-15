import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool
import time
import datetime

import hydra
import pandas as pd
from tqdm import tqdm
from glob import glob


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util

    util.create_directory_if_not_exists(pickle_dir)
        
    # unpickle the lenses from the population survey and create lens objects
    lens_dir = os.path.join('/data', 'bwedig', 'roman-population', 'data', 'lenses')
    lens_paths = glob(lens_dir + '/*')
    updated_lenses = []
    for i, lens in tqdm(enumerate(lens_paths), total=len(lens_paths)):
        lens = lens_util.unpickle_lens(lens, str(i).zfill(8))
        updated_lenses.append(lens)

    # pickle lens list
    pickle_target = os.path.join(pickle_dir, '01_skypy_output_lens_list')
    util.delete_if_exists(pickle_target)
    with open(pickle_target, 'ab') as results_file:
        pickle.dump(updated_lenses, results_file)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


if __name__ == '__main__':
    main()
