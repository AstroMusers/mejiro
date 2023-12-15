import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool
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

    # open pickled lens list
    with open(os.path.join(pickle_dir, '02_skypy_output_lens_list_with_subhalos'), 'rb') as results_file:
        lens_list = pickle.load(results_file) 

    # go sequentially
    dict_list = []
    # for i, lens in tqdm(enumerate(lens_list), total=len(lens_list)):
    #     lens, model = get_model(lens)
    #     np.save(os.path.join(array_dir, f'skypy_output_{str(i).zfill(8)}.npy'), model)
    #     updated_lenses.append(lens)

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    generator = util.batch_list(lens_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_model, batch):
            (updated_lens, model) = output
            lens_dict = {
                'lens': updated_lens,
                'model': model
            }
            dict_list.append(lens_dict)

    # pickle lens list
    pickle_target = os.path.join(pickle_dir, '03_skypy_output_lens_list_models')
    util.delete_if_exists(pickle_target)
    with open(pickle_target, 'ab') as results_file:
        pickle.dump(dict_list, results_file)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def get_model(lens):
    grid_oversample = 1
    return lens, lens.get_array(num_pix=51 * grid_oversample, side=5.61)


if __name__ == '__main__':
    main()
