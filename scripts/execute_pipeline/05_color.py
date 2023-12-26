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

    array_dir, repo_dir = config.machine.array_dir, config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # directory to write the output to
    output_dir = config.machine.dir_05
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # open pandeia arrays
    input_dir = config.machine.dir_04
    file_list = glob(input_dir + '/*.npy')
    num = int(len(file_list) / 4 - 100)  # TODO TEMP
    pandeia_list = []
    for i in range(num):
        f106 = np.load(input_dir + f'/pandeia_{str(i).zfill(8)}_f106.npy')
        f129 = np.load(input_dir + f'/pandeia_{str(i).zfill(8)}_f129.npy')
        f158 = np.load(input_dir + f'/pandeia_{str(i).zfill(8)}_f158.npy')
        f184 = np.load(input_dir + f'/pandeia_{str(i).zfill(8)}_f184.npy')
        rgb_tuple = (f106, f129, f184, output_dir, str(i).zfill(8))
        pandeia_list.append(rgb_tuple)

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(pandeia_list, process_count)
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
    # unpack tuple
    (f106, f129, f184, output_dir, uid) = input

    # generate and save color image
    from mejiro.helpers import color
    rgb_image = color.get_rgb(image_b=f106, image_g=f129, image_r=f184)
    np.save(os.path.join(output_dir, f'pandeia_color_{uid}.npy'), rgb_image)


if __name__ == '__main__':
    main()
