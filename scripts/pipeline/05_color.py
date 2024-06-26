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

    # build indices of uids
    lens_pickles = sorted(glob(config.machine.dir_02 + '/lens_with_subhalos_*'))
    lens_uids = [int(os.path.basename(i).split('_')[3].split('.')[0]) for i in lens_pickles]

    # implement limit, if applicable
    pipeline_params = util.hydra_to_dict(config.pipeline)
    limit = pipeline_params['limit']
    if limit is not None:
        lens_uids = lens_uids[:limit]

    # get rgb bands
    rgb_bands = pipeline_params['rgb_bands']
    assert len(rgb_bands) == 3, 'rgb_bands must be a list of 3 bands'

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    count = len(lens_uids)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    tuple_list = []
    for uid in lens_uids:
        tuple_list.append((uid, pipeline_params, input_dir, output_dir))

    # batch
    generator = util.batch_list(tuple_list, process_count)
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
    (uid, pipeline_params, input_dir, output_dir) = input
    rgb_bands = pipeline_params['rgb_bands']
    pieces = pipeline_params['pieces']

    # assign bands to colors
    red = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{rgb_bands[0]}.npy')
    green = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{rgb_bands[1]}.npy')
    blue = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{rgb_bands[2]}.npy')

    # generate and save color image
    from mejiro.helpers import color
    rgb_image = color.get_rgb(image_b=blue, image_g=green, image_r=red, stretch=4, Q=5)
    np.save(os.path.join(output_dir, f'galsim_color_{str(uid).zfill(8)}.npy'), rgb_image)

    if pieces:
        for piece in ['lens', 'source']:
            red = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{piece}_{rgb_bands[0]}.npy')
            green = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{piece}_{rgb_bands[1]}.npy')
            blue = np.load(input_dir + f'/galsim_{str(uid).zfill(8)}_{piece}_{rgb_bands[2]}.npy')
            rgb_image = color.get_rgb(image_b=blue, image_g=green, image_r=red, stretch=4, Q=5)
            np.save(os.path.join(output_dir, f'galsim_color_{str(uid).zfill(8)}_{piece}.npy'), rgb_image)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    return execution_time


if __name__ == '__main__':
    main()
