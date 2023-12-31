import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # directory to write the output to
    output_dir = config.machine.dir_04  # os.path.join(config.machine.pipeline_dir, '04_test')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # TODO unpickling at beginning means all of this is held in memory, so unpickle over time

    # open pickled lens dict list
    f106_list = util.unpickle_all(config.machine.dir_03,
                                  prefix='lens_dict_*_f106')  # os.path.join(config.machine.pipeline_dir, '03_test')
    f129_list = util.unpickle_all(config.machine.dir_03, prefix='lens_dict_*_f129')
    f184_list = util.unpickle_all(config.machine.dir_03, prefix='lens_dict_*_f184')
    dict_list = []
    for i, _ in enumerate(f106_list):
        dict_list.append(f106_list[i])
        dict_list.append(f129_list[i])
        dict_list.append(f184_list[i])
    # dict_list = util.unpickle_all(config.machine.dir_03)
        
    # TODO TEMP: limit list
    dict_list = dict_list[:100]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for i, _ in enumerate(dict_list):
        tuple_list.append((dict_list[i], pipeline_params, output_dir))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    execution_times = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_image, batch):
            execution_times.append(output)

    # TODO update and append results from each batch, instead of writing all at end
    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image(input):
    from mejiro.helpers import pandeia_input

    # unpack tuple
    (lens_dict, pipeline_params, output_dir) = input

    # unpack lens_dict
    array = lens_dict['model']
    lens = lens_dict['lens']
    uid = lens.uid
    band = lens.band

    # unpack pipeline_params
    max_scene_size = pipeline_params['max_scene_size']
    num_samples = pipeline_params['num_samples']

    # build Pandeia input
    calc, _ = pandeia_input.build_pandeia_calc(array, lens, background=True, noise=True, band=band, max_scene_size=max_scene_size,
                                               num_samples=num_samples, suppress_output=True)

    # generate Pandeia image and save
    image, execution_time = pandeia_input.get_pandeia_image(calc, suppress_output=True)
    np.save(os.path.join(output_dir, f'pandeia_{uid}_{band}.npy'), image)

    return execution_time


if __name__ == '__main__':
    main()
