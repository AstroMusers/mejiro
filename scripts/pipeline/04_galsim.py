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

    # build indices of uids
    lens_pickles = sorted(glob(config.machine.dir_02 + '/lens_with_subhalos_*.pkl'))
    lens_uids = [int(os.path.basename(i).split('_')[3].split('.')[0]) for i in lens_pickles]

    # implement limit, if applicable
    pipeline_params = util.hydra_to_dict(config.pipeline)
    limit = pipeline_params['limit']
    if limit is not None:
        lens_uids = lens_uids[:limit]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    # TODO for some reason, this particular script needs more headroom cores. maybe it's a memory thing?
    process_count -= 36
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
            execution_times.extend(output)

    np.save(os.path.join(array_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image(input):
    from mejiro.helpers import gs
    from mejiro.utils import util

    # unpack tuple
    (uid, pipeline_params, input_dir, output_dir) = input

    # unpack pipeline_params
    bands = pipeline_params['bands']
    grid_oversample = pipeline_params['grid_oversample']
    exposure_time = pipeline_params['exposure_time']
    suppress_output = pipeline_params['suppress_output']
    final_pixel_side = pipeline_params['final_pixel_side']
    num_pix = pipeline_params['num_pix']
    # seed = pipeline_params['seed']  # TODO think about what this is doing
    pieces = pipeline_params['pieces']

    # load lens
    lens = util.unpickle(os.path.join(input_dir, f'lens_{str(uid).zfill(8)}.pkl'))

    # load the appropriate arrays
    arrays = [np.load(f'{input_dir}/array_{lens.uid}_{band}.npy') for band in bands]
    if pieces:
        lens_surface_brightness = [np.load(f'{input_dir}/array_{lens.uid}_lens_{band}.npy') for band in bands]
        source_surface_brightness = [np.load(f'{input_dir}/array_{lens.uid}_source_{band}.npy') for band in bands]
        pieces_args = {'lens_surface_brightness': lens_surface_brightness, 'source_surface_brightness': source_surface_brightness}
    else:
        pieces_args = {}

    # determine detector and position
    detector = gs.get_random_detector(suppress_output)
    detector_pos = gs.get_random_detector_pos(input_size=num_pix, suppress_output=suppress_output)

    gs_results = gs.get_images(lens, 
                                            arrays, 
                                            bands, 
                                            input_size=num_pix, 
                                            output_size=final_pixel_side,
                                            grid_oversample=grid_oversample, psf_oversample=grid_oversample,
                                            *pieces_args,
                                            detector=detector,
                                            detector_pos=detector_pos, 
                                            exposure_time=exposure_time, 
                                            ra=None, 
                                            dec=None,
                                            seed=random.randint(0, 2 ** 16 - 1), 
                                            validate=False, 
                                            suppress_output=suppress_output)

    if pieces:
        results, lenses, sources, execution_time = gs_results
        results += lenses
        results += sources
    else:
        results, execution_time = gs_results

    j = 0
    for i, (band, result) in enumerate(zip(bands, results)):
        if j == 0:
            np.save(os.path.join(output_dir, f'galsim_{lens.uid}_{band}.npy'), result)
        elif j == 1:
            np.save(os.path.join(output_dir, f'galsim_{lens.uid}_lens_{band}.npy'), result)
        elif j == 2:
            np.save(os.path.join(output_dir, f'galsim_{lens.uid}_source_{band}.npy'), result)
        if i % len(set(bands)) == len(set(bands)) - 1:
            j += 1

    return execution_time


if __name__ == '__main__':
    main()
