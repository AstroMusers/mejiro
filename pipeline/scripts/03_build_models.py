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

    # get directories
    repo_dir = config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # directory to get pickled lenses (with subhalos) from
    input_dir = config.machine.dir_02

    # directory to write the output to
    output_dir = config.machine.dir_03
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
    # TODO having resource issues here as well
    process_count -= 10
    count = len(lens_uids)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    tuple_list = []
    for i in lens_uids:
        tuple_list.append((i, pipeline_params, input_dir, output_dir))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(get_model, batch)

    stop = time.time()
    util.print_execution_time(start, stop)


def get_model(input):
    from mejiro.utils import util

    # unpack tuple
    (i, pipeline_params, input_dir, output_dir) = input

    # unpack pipeline params
    bands = pipeline_params['bands']
    num_pix = pipeline_params['num_pix']
    side = pipeline_params['side']
    grid_oversample = pipeline_params['grid_oversample']
    pieces = pipeline_params['pieces']

    # load the lens based on uid
    lens = util.unpickle(os.path.join(input_dir, f'lens_with_subhalos_{str(i).zfill(8)}.pkl'))

    # generate lenstronomy model and save
    for band in bands:
        if pieces:
            model, lens_surface_brightness, source_surface_brightness = lens.get_array(
                num_pix=num_pix * grid_oversample, side=side, band=band, return_pieces=True)
            np.save(os.path.join(output_dir, f'array_{lens.uid}_lens_{band}'), lens_surface_brightness)
            np.save(os.path.join(output_dir, f'array_{lens.uid}_source_{band}'), source_surface_brightness)
        else:
            model = lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
        np.save(os.path.join(output_dir, f'array_{lens.uid}_{band}'), model)

    # pickle lens to save attributes updated by get_array()
    pickle_target_lens = os.path.join(output_dir, f'lens_{lens.uid}.pkl')
    util.pickle(pickle_target_lens, lens)


if __name__ == '__main__':
    main()
