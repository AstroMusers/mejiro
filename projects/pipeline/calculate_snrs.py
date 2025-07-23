import argparse
import os
import time
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from mejiro.exposure import Exposure
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper


PREV_SCRIPT_NAME = '05'
SCRIPT_NAME = 'snr'
SUPPORTED_INSTRUMENTS = ['roman', 'hwo']


def main(args):
    start = time.time()

    # ensure the configuration file has a .yaml or .yml extension
    if not args.config.endswith(('.yaml', '.yml')):
        if os.path.exists(args.config + '.yaml'):
            args.config += '.yaml'
        elif os.path.exists(args.config + '.yml'):
            args.config += '.yml'
        else:
            raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")

    # read configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    snr_config = config['snr']

    # initialize PipeLineHelper
    pipeline = PipelineHelper(config, PREV_SCRIPT_NAME, SCRIPT_NAME)

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='Exposure', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name == 'hwo':
        input_pickles = pipeline.retrieve_hwo_pickles(prefix='Exposure', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        if pipeline.verbose: print(f'Limiting to {pipeline.limit} lens(es)')
        input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        if pipeline.limit < count:
            count = pipeline.limit
    if pipeline.verbose: print(f'Processing {count} lens(es)')

    # tuple the parameters
    tuple_list = [(pipeline, snr_config, input_pickle) for input_pickle in input_pickles]

    # process the tasks with ProcessPoolExecutor
    name_snr_pairs = []
    with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
        futures = [executor.submit(calculate_snr, task) for task in tuple_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            name_snr_pairs.append(future.result())

    output_path = os.path.join(pipeline.output_dir, 'name_snr_pairs.pkl')
    util.pickle(output_path, name_snr_pairs)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def calculate_snr(input):
    # unpack tuple
    (pipeline, snr_config, input_pickle) = input

    # unpack snr config
    snr_per_pixel_threshold = snr_config['snr_per_pixel_threshold']

    # load exposure
    exposure = util.unpickle(input_pickle)

    # calculate SNR
    snr = exposure.get_snr(snr_per_pixel_threshold=snr_per_pixel_threshold)

    return (exposure.synthetic_image.strong_lens.name, snr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and cache Roman PSFs.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
