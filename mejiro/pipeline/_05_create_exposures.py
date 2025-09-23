"""
Generates exposures from synthetic images, i.e., apply sky background and detector effects to idealized images.

This script processes synthetic images produced in previous pipeline steps, generating exposures for each lensing system using instrument-specific parameters and simulation engines. It reads a mejiro YAML configuration file specifying exposure options. Multiprocessing is used to parallelize exposure creation across available CPU cores.

Usage:
    python3 _05_create_exposures.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from mejiro.exposure import Exposure
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper


PREV_SCRIPT_NAME = '04'
SCRIPT_NAME = '05'
SUPPORTED_INSTRUMENTS = ['roman', 'hwo']


def main(args):
    start = time.time()

    # initialize PipeLineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME)

    # retrieve configuration parameters
    imaging_config = pipeline.config['imaging']

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='SyntheticImage', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name == 'hwo':
        input_pickles = pipeline.retrieve_hwo_pickles(prefix='SyntheticImage', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        if pipeline.verbose: print(f'Limiting to {pipeline.limit} image(s)')
        input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        if pipeline.limit < count:
            count = pipeline.limit
    if pipeline.verbose: print(f'Processing {count} image(s)')

    # tuple the parameters
    tuple_list = [(pipeline, imaging_config, input_pickle) for input_pickle in input_pickles]

    # process the tasks with ProcessPoolExecutor
    execution_times = []
    with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
        futures = [executor.submit(get_image, task) for task in tuple_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            execution_times.append(future.result())

    np.save(os.path.join(pipeline.output_dir, 'execution_times.npy'), execution_times)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def get_image(input):
    # unpack tuple
    (pipeline, imaging_config, input_pickle) = input

    # unpack imaging config
    exposure_time = imaging_config['exposure_time']
    engine = imaging_config['engine']
    engine_params = imaging_config['engine_params']

    # load synthetic image
    synthetic_image = util.unpickle(input_pickle)
        
    if pipeline.instrument_name == 'roman':
        sca_string = roman_util.get_sca_string(synthetic_image.instrument_params['detector']).lower()
        output_dir = os.path.join(pipeline.output_dir, sca_string)
    else:
        output_dir = pipeline.output_dir

    # build and pickle exposure
    try:
        exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine=engine,
                        engine_params=engine_params,
                        verbose=False)
        util.pickle(os.path.join(output_dir, f'Exposure_{synthetic_image.strong_lens.name}_{synthetic_image.band}.pkl'), exposure)
    except Exception as e:
        failed_pickle_path = os.path.join(output_dir, f'failed_{synthetic_image.strong_lens.name}_{synthetic_image.band}.pkl')
        util.pickle(failed_pickle_path, synthetic_image)
        print(f'Error creating synthetic image for lens {synthetic_image.strong_lens.name} in band {synthetic_image.band}: {e}. Pickling to {failed_pickle_path}')
        return

    return exposure.calc_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate exposures.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
