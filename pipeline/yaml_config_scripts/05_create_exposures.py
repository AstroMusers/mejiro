import json
import multiprocessing
import os
import random
import sys
import time
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


PREV_SCRIPT_NAME = '04'
SCRIPT_NAME = '05'


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util

    # retrieve configuration parameters
    dev = config['pipeline']['dev']
    verbose = config['pipeline']['verbose']
    data_dir = config['data_dir']
    limit = config['pipeline']['limit']
    imaging_config = config['pipeline']['survey']['imaging']

    # set nice level
    os.nice(config['pipeline']['nice'])

    # set up top directory for all pipeline output
    if dev:
        pipeline_dir = os.path.join(data_dir, 'pipeline_dev')
    else:
        pipeline_dir = os.path.join(data_dir, 'pipeline')
    util.create_directory_if_not_exists(pipeline_dir)

    # tell script where the output of previous script is
    input_dir = os.path.join(pipeline_dir, PREV_SCRIPT_NAME)

    # parse output of previous script to determine which SCAs to process
    sca_dir_names = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dir_names])
    scas = [str(sca).zfill(2) for sca in scas]

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    output_sca_dirs = []
    for sca in scas:
        sca_dir = os.path.join(output_dir, f'sca{sca}')
        os.makedirs(sca_dir, exist_ok=True)
        output_sca_dirs.append(sca_dir)
    if verbose: print(f'Set up output directories {output_sca_dirs}')  

    # set PSF cache dir
    psf_cache_dir = os.path.join(data_dir, 'cached_psfs')
    if verbose: print(f'Retrieving PSFs from {psf_cache_dir}')

    # get lens UIDs, organized by SCA
    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_dir + f'/sca{sca}/SyntheticImage_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[1] for i in pickled_lenses]
        lens_uids = list(set(lens_uids))  # remove duplicates, e.g., if multiple bands
        lens_uids = sorted(lens_uids)
        uid_dict[sca] = lens_uids  # for each SCA, list of UIDs of associated lenses

    # count total lenses
    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)
    if verbose: print(f'Processing {count} lens(es)')

    # if a limit is in place and there are more lenses than the limit, limit
    if limit is not None and limit < count:
        if verbose: print(f'Limiting to {limit} lens(es)')
        count = limit

    # determine number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)
    process_count -= config['pipeline']['headroom_cores']['script_05']
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_sca_dir = os.path.join(input_dir, f'sca{sca}')
        output_sca_dir = os.path.join(output_dir, f'sca{sca}')

        for uid in lens_uids:
            tuple_list.append((uid, imaging_config, input_sca_dir, output_sca_dir, psf_cache_dir))
            i += 1
            if i == limit:
                break

    # process the tasks with ProcessPoolExecutor
    execution_times = []
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(get_image, task) for task in tuple_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            execution_times.extend(future.result())

    np.save(os.path.join(output_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline_dir, 'execution_times.json'))


def get_image(input):
    from mejiro.exposure import Exposure
    from mejiro.utils import util

    # unpack tuple
    (uid, imaging_config, input_sca_dir, output_sca_dir, psf_cache_dir) = input

    # unpack imaging config
    exposure_time = imaging_config['exposure_time']
    engine = imaging_config['engine']
    engine_params = imaging_config['engine_params']

    # load synthetic image for each band
    synthetic_images = []
    filepaths = sorted(glob(os.path.join(input_sca_dir, f'SyntheticImage_{uid}_*.pkl')))
    for f in filepaths:
        try:
            synthetic_image = util.unpickle(f)
            synthetic_images.append(synthetic_image)
        except Exception as e:
            print(f'Error unpickling SyntheticImage {uid}: {e}')
            return 0
    
    # create exposures
    calc_times = []
    for synth in synthetic_images:
        exposure = Exposure(synthetic_image,
                            exposure_time=exposure_time,
                            engine=engine,
                            engine_params=engine_params,
                            psf=None,
                            verbose=False,
                            check_cache=True,
                            psf_cache_dir=psf_cache_dir)
        calc_times.append(exposure.calc_time)
    
        # pickle exposure
        util.pickle(os.path.join(output_sca_dir, f'Exposure_{uid}_{synth.band}.pkl'), exposure)

    return calc_times


if __name__ == '__main__':
    main()
