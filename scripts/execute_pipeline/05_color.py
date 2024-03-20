import datetime
import multiprocessing
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
    input_dir = config.machine.dir_04

    # directory to write the output to
    output_dir = config.machine.dir_05
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # list uids and build input list
    # TODO LIMIT IS TEMP
    # limit = 9
    lens_pickles = glob(config.machine.dir_02 + '/lens_with_subhalos_*')
    count = len(lens_pickles)
    input_list = [(str(uid).zfill(8), input_dir, output_dir) for uid in list(range(count))]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(input_list, process_count)
    batches = list(generator)

    # process the batches
    execution_times = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for output in pool.map(get_image, batch):
            execution_times.append(output)

    # TODO update and append results from each batch, instead of writing all at end; or maybe this is fine for the number of images we'll be generating
    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image(input):
    start = time.time()

    # unpack tuple
    (uid, input_dir, output_dir) = input

    f106 = np.load(input_dir + f'/galsim_{uid}_F106.npy')
    f129 = np.load(input_dir + f'/galsim_{uid}_F129.npy')
    f184 = np.load(input_dir + f'/galsim_{uid}_F184.npy')

    # generate and save color image
    from mejiro.helpers import color
    rgb_image = color.get_rgb(image_b=f106, image_g=f129, image_r=f184, stretch=4, Q=5)
    np.save(os.path.join(output_dir, f'galsim_color_{uid}.npy'), rgb_image)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    return execution_time


if __name__ == '__main__':
    main()
