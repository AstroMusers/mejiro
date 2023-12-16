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

    # directory to write the output to
    output_dir = os.path.join(array_dir, '04_pandeia_output')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # open pickled lens dict list
    input_dir = os.path.join(pickle_dir, '03_models_and_updated_lenses')
    dict_list = util.unpickle_all(os.path.join(input_dir), prefix='lens_dict_')

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for i, _ in enumerate(dict_list):
        tuple_list.append((dict_list[i], pipeline_params))

    # batch
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
    util.print_execution_time(start, stop)


def get_image(input):
    # unpack tuple
    (lens_dict, pipeline_params) = input

    # unpack lens_dict
    array = lens_dict['model']
    lens = lens_dict['lens']
    uid = lens.uid

    # unpack pipeline_params
    max_scene_size = pipeline_params['max_scene_size']
    num_samples = pipeline_params['num_samples']

    from mejiro.helpers import pandeia_input

    calc, _ = pandeia_input.build_pandeia_calc(array, lens, max_scene_size=max_scene_size, num_samples=num_samples, suppress_output=True)

    image, execution_time = pandeia_input.get_pandeia_image(calc, suppress_output=True)

    return uid, image, execution_time


if __name__ == '__main__':
    main()
