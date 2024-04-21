import multiprocessing
import random
import os
import sys
import time
from glob import glob
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

    # directory to read from
    input_dir = config.machine.dir_03

    # directory to write the output to
    output_dir = config.machine.dir_04
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # list uids
    # TODO LIMIT IS TEMP
    # limit = 9
    # uid_list = list(range(limit))
    # count number of lenses and build indices of uids
    lens_pickles = glob(config.machine.dir_02 + '/lens_with_subhalos_*')
    count = len(lens_pickles)
    uid_list = list(range(count))

    # get bands
    bands = util.hydra_to_dict(config.pipeline)['band']

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for uid in uid_list:
        tuple_list.append((uid, pipeline_params, input_dir, output_dir, bands))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    execution_times = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_image, batch):
            execution_times.extend(output)

    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image(input):
    from mejiro.helpers import gs
    from mejiro.utils import util

    # unpack tuple
    (uid, pipeline_params, input_dir, output_dir, bands) = input

    # unpack pipeline_params
    grid_oversample = pipeline_params['grid_oversample']
    exposure_time = pipeline_params['exposure_time']
    suppress_output = pipeline_params['suppress_output']
    final_pixel_side = pipeline_params['final_pixel_side']
    num_pix = pipeline_params['num_pix']
    # seed = pipeline_params['seed']  # TODO think about what this is doing

    # load lens
    lens = util.unpickle(os.path.join(input_dir, f'lens_{str(uid).zfill(8)}'))

    # load the appropriate arrays
    arrays = [np.load(f'{input_dir}/array_{lens.uid}_{band}.npy') for band in bands]

    # determine detector and position
    detector = gs.get_random_detector(suppress_output)
    detector_pos = gs.get_random_detector_pos(input_size=num_pix, suppress_output=suppress_output)

    results, execution_time = gs.get_images(lens, arrays, bands, input_size=num_pix, output_size=final_pixel_side,
                                            grid_oversample=grid_oversample, psf_oversample=grid_oversample,
                                            detector=detector,
                                            detector_pos=detector_pos, exposure_time=exposure_time, ra=None, dec=None,
                                            seed=random.randint(0, 2 ** 16 - 1), validate=False, suppress_output=suppress_output)

    for band, result in zip(bands, results):
        np.save(os.path.join(output_dir, f'galsim_{lens.uid}_{band}.npy'), result)

    return execution_time


if __name__ == '__main__':
    main()
