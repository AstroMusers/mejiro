import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool

import hydra
import numpy as np
from tqdm import tqdm

from mejiro.utils import util


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir
    array_dir = os.path.join(array_dir, 'skypy_output')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # open pickled lens list
    with open(os.path.join(pickle_dir, '02_skypy_output_lens_list_with_subhalos'), 'rb') as results_file:
        lens_list = pickle.load(results_file)

    updated_lens_list = []

    # go sequentially so that lens_list is modified
    grid_oversample = 5
    for i, lens in tqdm(enumerate(lens_list), total=len(lens_list)):
        model = lens.get_array(num_pix=97 * grid_oversample, side=10.67)
        np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(8)}.npy'), model)
        updated_lens_list.append(lens)

    # # split up the lenses into batches based on core count
    # cpu_count = multiprocessing.cpu_count()
    # process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    # generator = util.batch_list(lens_list, process_count)
    # batches = list(generator)

    # # process the batches
    # i = 0
    # for batch in tqdm(batches):
    #     pool = Pool(processes=process_count)
    #     for output in pool.map(get_model, batch):
    #         (model) = output
    #         np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(8)}.npy'), model)
    #         i += 1

    # pickle lens list
    pickle_target = os.path.join(pickle_dir, '03_skypy_output_lens_list_models')
    util.delete_if_exists(pickle_target)
    with open(pickle_target, 'ab') as results_file:
        pickle.dump(updated_lens_list, results_file)


def get_model(lens):
    grid_oversample = 5
    return lens.get_array(num_pix=97 * grid_oversample, side=10.67)


if __name__ == '__main__':
    main()