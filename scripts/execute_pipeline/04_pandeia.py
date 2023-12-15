import multiprocessing
import os
import sys
from glob import glob
from multiprocessing import Pool
import pickle
import time
import datetime

import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    array_dir = os.path.join(array_dir, 'skypy_output')
    util.create_directory_if_not_exists(array_dir)
    util.clear_directory(array_dir)

    # open pickled lens dict list
    with open(os.path.join(pickle_dir, '03_skypy_output_lens_list_models'), 'rb') as results_file:
        dict_list = pickle.load(results_file) 

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    generator = util.batch_list(dict_list, process_count)
    batches = list(generator)

    # process the batches
    execution_times = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_image, batch):
            (uid, image, execution_time) = output
            np.save(os.path.join(array_dir, f'pandeia_{uid}.npy'), image)
            execution_times.append(execution_time)

    # TODO update and append results from each batch, instead of writing all at end
    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def get_image(lens_dict):
    # unpack lens_dict
    array = lens_dict['model']
    lens = lens_dict['lens']
    uid = lens.uid

    from mejiro.helpers import pandeia_input

    calc, _ = pandeia_input.build_pandeia_calc(array, lens, num_samples=10000, suppress_output=True)

    image, execution_time = pandeia_input.get_pandeia_image(calc, suppress_output=True)

    return uid, image, execution_time


if __name__ == '__main__':
    main()
