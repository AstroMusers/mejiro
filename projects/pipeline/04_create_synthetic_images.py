import argparse
import multiprocessing
import os
import random
import time
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper


PREV_SCRIPT_NAME = '03'  # '03'
SCRIPT_NAME = '04'
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
    synthetic_image_config = config['synthetic_image']
    psf_config = config['psf']

    # initialize PipeLineHelper
    pipeline = PipelineHelper(config, PREV_SCRIPT_NAME, SCRIPT_NAME)

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='lens', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name == 'hwo':
        input_pickles = pipeline.retrieve_hwo_pickles(prefix='lens', suffix='', extension='.pkl')
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
    tuple_list = [(pipeline, synthetic_image_config, psf_config, input_pickle) for input_pickle in input_pickles]

    # define the number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count
    process_count -= config['headroom_cores']['script_04']
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # submit tasks to the executor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(create_synthetic_image, task): task for task in tuple_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def create_synthetic_image(input):
    # unpack tuple
    (pipeline, synthetic_image_config, psf_config, input_pickle) = input

    # unpack pipeline params
    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    pieces = synthetic_image_config['pieces']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    num_pix = psf_config['num_pixes'][0]
    divide_up_detector = psf_config.get('divide_up_detector')  # Roman-specific parameter, not required for HWO

    # unpickle the lens
    lens = util.unpickle(input_pickle)

    # build kwargs_numerics
    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode
    }

    get_psf_args = {}
    instrument_params = {}

    # set detector and pick random position
    if pipeline.instrument_name == 'roman':
        possible_detector_positions = roman_util.divide_up_sca(divide_up_detector)
        detector_position = random.choice(possible_detector_positions)
        instrument_params['detector'] = pipeline.parse_sca_from_filename(input_pickle)
        instrument_params['detector_position'] = detector_position
        get_psf_args |= instrument_params

        sca_string = roman_util.get_sca_string(instrument_params['detector'])
        output_dir = os.path.join(pipeline.output_dir, sca_string)
    else:
        output_dir = pipeline.output_dir

    # generate synthetic images
    for band in bands:
        # get PSF
        get_psf_args |= {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
            'verbose': False,
        }
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

        # build and pickle synthetic image
        try:
            synthetic_image = SyntheticImage(strong_lens=lens,
                                        instrument=pipeline.instrument,
                                        band=band,
                                        fov_arcsec=fov_arcsec,
                                        instrument_params=instrument_params,
                                        kwargs_numerics=kwargs_numerics,
                                        kwargs_psf=kwargs_psf,
                                        pieces=pieces,
                                        verbose=False)
            util.pickle(os.path.join(output_dir, f'SyntheticImage_{lens.name}_{band}.pkl'), synthetic_image)
        except Exception as e:
            failed_pickle_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
            util.pickle(failed_pickle_path, lens)
            print(f'Error creating synthetic image for lens {lens.name} in band {band}: {e}. Pickling to {failed_pickle_path}')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic images")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
