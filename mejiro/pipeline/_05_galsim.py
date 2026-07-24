"""
Generates exposures from synthetic images, i.e., apply sky background and detector effects to idealized images.

This script processes synthetic images produced in previous pipeline steps, generating exposures for each lensing system using instrument-specific parameters and simulation engines. It reads a mejiro YAML configuration file specifying exposure options. Multiprocessing is used to parallelize exposure creation across available CPU cores.

Usage:
    python3 _05_galsim.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from tqdm import tqdm

import logging

from mejiro.exposure import Exposure
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)

PREV_SCRIPT_NAME = '04'
SCRIPT_NAME = '05_galsim'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    # initialize PipelineHelper (we handle the default wipe ourselves so we can count + warn first)
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS,
                              delete_existing_output=False)

    if not args.resume:
        existing = [p for p in glob(os.path.join(pipeline.output_dir, '**', '*'), recursive=True)
                    if os.path.isfile(p)]
        if existing:
            logger.warning(
                f'Deleting {len(existing)} existing output file(s) in '
                f'{pipeline.output_dir} and rebuilding from scratch. Pass --resume to keep them.'
            )
            util.clear_directory(pipeline.output_dir)

    # retrieve configuration parameters
    imaging_config = pipeline.config['imaging']

    # imaging.serialization selects how the Exposure OUTPUT is written: 'lightweight'
    # writes a compact .npz (data + optional pieces + JSON metadata) via
    # Exposure.save_lightweight; 'full' pickles the whole Exposure object. This is
    # distinct from synthetic_image.serialization below, which governs the step-04 INPUT.
    output_serialization = imaging_config['serialization']
    if output_serialization not in ('full', 'lightweight'):
        raise ValueError(
            f"imaging.serialization must be 'full' or 'lightweight', got {output_serialization!r}"
        )
    output_ext = '.npz' if output_serialization == 'lightweight' else '.pkl'

    # The galsim engine needs the full SyntheticImage (lens model, PSF, pixel
    # grid, etc.), so lightweight inputs from step 04 are not usable here.
    serialization = pipeline.config['synthetic_image']['serialization']
    if serialization == 'lightweight':
        raise ValueError(
            "_05_galsim requires the full SyntheticImage but "
            "synthetic_image.serialization is set to 'lightweight'. Either "
            "re-run step 04 with serialization: full, or use the romanisim "
            "path (_05_romanisim.py) which is compatible with lightweight."
        )

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='SyntheticImage', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name == 'hwo' or pipeline.instrument_name == 'jwst':
        input_pickles = pipeline.retrieve_pickles(prefix='SyntheticImage', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # resume: skip inputs whose Exposure output already exists. mirror the input's directory
    # layout (per-SCA for roman, flat otherwise) to derive the expected output path.
    if args.resume:
        def _expected_output(input_pickle):
            base = os.path.basename(input_pickle)
            assert base.startswith('SyntheticImage_') and base.endswith('.pkl'), base
            out_name = 'Exposure_' + base[len('SyntheticImage_'):-len('.pkl')] + output_ext
            if pipeline.instrument_name == 'roman':
                sca_dir = os.path.basename(os.path.dirname(input_pickle))
                return os.path.join(pipeline.output_dir, sca_dir, out_name)
            return os.path.join(pipeline.output_dir, out_name)

        total_before = len(input_pickles)
        input_pickles = [p for p in input_pickles if not os.path.exists(_expected_output(p))]
        skipped = total_before - len(input_pickles)
        logger.info(
            f'Resuming: {skipped} of {total_before} image(s) already complete, '
            f'{len(input_pickles)} remaining.'
        )

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} image(s)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} image(s)')

    # tuple the parameters
    tuple_list = [(pipeline, imaging_config, input_pickle) for input_pickle in input_pickles]

    # process the tasks with ProcessPoolExecutor
    execution_times = []
    try:
        with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
            futures = [executor.submit(get_image, task) for task in tuple_list]

            for future in tqdm(as_completed(futures), total=len(futures)):
                execution_times.append(future.result())
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

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
    output_serialization = imaging_config['serialization']
    output_ext = '.npz' if output_serialization == 'lightweight' else '.pkl'

    # load synthetic image
    synthetic_image = util.unpickle(input_pickle)

    if pipeline.instrument_name == 'roman':
        sca_string = roman_util.get_sca_string(synthetic_image.instrument_params['detector']).lower()
        output_dir = os.path.join(pipeline.output_dir, sca_string)
    else:
        output_dir = pipeline.output_dir

    # build and save exposure ('lightweight' -> compact .npz, 'full' -> whole-object pickle)
    try:
        exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine=engine,
                        engine_params=engine_params)
        output_path = os.path.join(output_dir, f'Exposure_{synthetic_image.strong_lens.name}_{synthetic_image.band}{output_ext}')
        if output_serialization == 'lightweight':
            exposure.save_lightweight(output_path)
        else:
            util.pickle(output_path, exposure)
    except Exception as e:
        failed_pickle_path = os.path.join(output_dir, f'failed_{synthetic_image.strong_lens.name}_{synthetic_image.band}.pkl')
        util.pickle(failed_pickle_path, synthetic_image)
        logger.warning(f'Error creating synthetic image for lens {synthetic_image.strong_lens.name} in band {synthetic_image.band}: {e}. Pickling to {failed_pickle_path}')
        return

    return exposure.calc_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate exposures.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
