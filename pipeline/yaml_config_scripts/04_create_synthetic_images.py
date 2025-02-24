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


PREV_SCRIPT_NAME = '03'
SCRIPT_NAME = '04'


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.instruments.roman import Roman
    from mejiro.utils import util

    # retrieve configuration parameters
    dev = config['pipeline']['dev']
    verbose = config['pipeline']['verbose']
    data_dir = config['data_dir']
    limit = config['pipeline']['limit']
    synthetic_image_config = config['pipeline']['survey']['synthetic_image']

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
    input_sca_dirs = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in input_sca_dirs])
    scas = [str(sca).zfill(2) for sca in scas]
    if verbose: print(f'Reading from {input_sca_dirs}')

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

    # build instrument
    roman = Roman()

    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_dir + f'/sca{sca}/lens_with_subhalos_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[3].split('.')[0] for i in pickled_lenses]
        uid_dict[sca] = lens_uids

    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)

    if limit is not None and limit < count:
        count = limit

    # tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_sca_dir = os.path.join(input_dir, f'sca{sca}')
        output_sca_dir = os.path.join(output_dir, f'sca{sca}')

        for uid in lens_uids:
            tuple_list.append((uid, sca, roman, synthetic_image_config, input_sca_dir, output_sca_dir))
            i += 1
            if i == limit:
                break
        else:
            continue
        break

    # Define the number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count
    process_count -= config['pipeline']['headroom_cores']['script_04']
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # Submit tasks to the executor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(create_synthetic_image, task): task for task in tuple_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline_dir, 'execution_times.json'))


def create_synthetic_image(input):
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils import roman_util, util

    # unpack tuple
    (uid, sca, roman, synthetic_image_config, input_dir, output_dir) = input

    # unpack pipeline params
    bands = synthetic_image_config['bands']
    side = synthetic_image_config['side']
    pieces = synthetic_image_config['pieces']
    base_supersampling = synthetic_image_config['base_supersampling']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    divide_up_sca = synthetic_image_config['divide_up_sca']

    # load the lens based on uid
    lens = util.unpickle(os.path.join(input_dir, f'lens_with_subhalos_{uid}.pkl'))
    assert lens.uid == uid, f'UID mismatch: {lens.uid} != {uid}'

    # build kwargs_numerics
    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode
    }

    # set detector and pick random position
    possible_detector_positions = roman_util.divide_up_sca(divide_up_sca)
    detector_pos = random.choice(possible_detector_positions)
    instrument_params = {
        'detector': int(sca),
        'detector_position': detector_pos
    }

    # generate synthetic images
    for band in bands:
        synthetic_image = SyntheticImage(strong_lens=lens,
                                     instrument=roman,
                                     band=band,
                                     arcsec=side,
                                     oversample=base_supersampling,
                                     kwargs_numerics=kwargs_numerics,
                                     pieces=pieces,
                                     verbose=False,
                                     instrument_params=instrument_params)
        util.pickle(os.path.join(output_dir, f'SyntheticImage_{uid}_{band}.pkl'), synthetic_image)


if __name__ == '__main__':
    main()
