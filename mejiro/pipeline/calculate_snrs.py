"""
Calculates signal-to-noise ratios (SNRs) for simulated exposures.

This script computes SNR values for each lensing system processed in previous
pipeline steps and saves name-SNR pairs for downstream filtering or analysis.
It reads a YAML configuration file specifying SNR calculation parameters and
supports both sequential and parallel processing modes.

Usage:
    python3 calculate_snrs.py --config <config.yaml> [--sequential] [--resume]

Arguments:
    --config: Path to the YAML configuration file.
    --sequential: Run in sequential mode instead of parallel.
    --resume: Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.
"""

import argparse
import os
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from tqdm import tqdm

import logging

from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)

PREV_SCRIPT_NAME = '05_romanisim'
SCRIPT_NAME = 'snr'
SUPPORTED_INSTRUMENTS = ['roman', 'hwo']


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    # initialize PipelineHelper (we handle the default wipe ourselves so we can count + warn first)
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS,
                              delete_existing_output=False)

    output_path = os.path.join(pipeline.output_dir, 'name_snr_pairs.pkl')

    if not args.resume:
        if os.path.exists(output_path):
            logger.warning(
                f'Deleting existing output file {os.path.basename(output_path)} and '
                f'rebuilding from scratch. Pass --resume to keep it.'
            )
            os.remove(output_path)
    elif os.path.exists(output_path):
        logger.info(f'Output already exists at {output_path}; skipping.')
        return

    # retrieve configuration parameters
    snr_config = pipeline.config['snr']

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='Exposure', suffix='', extension='.npy')
    elif pipeline.instrument_name == 'hwo':
        input_pickles = pipeline.retrieve_pickles(prefix='Exposure', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} exposure(s)')

    # tuple the parameters
    tuple_list = [(pipeline, snr_config, input_pickle) for input_pickle in input_pickles]

    # process the tasks with ProcessPoolExecutor
    name_snr_pairs = []
    try:
        with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
            futures = [executor.submit(calculate_snr, task) for task in tuple_list]

            for future in tqdm(as_completed(futures), total=len(futures)):
                name_snr_pairs.append(future.result())
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    util.pickle(output_path, name_snr_pairs)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def calculate_snr(input):
    # unpack tuple
    (pipeline, snr_config, input_pickle) = input

    # parse pickle filename to get system name and band
    basename = os.path.basename(input_pickle)
    parts = basename.split('_')
    system_name = "_".join(parts[1:-1])
    band = parts[-1].rsplit('.', 1)[0]

    # unpack snr config
    snr_per_pixel_threshold = snr_config['snr_per_pixel_threshold']

    # load exposure
    # exposure = util.unpickle(input_pickle)
    exposure = None

    # calculate SNR
    # try:
    #     snr = exposure.get_snr(snr_per_pixel_threshold=snr_per_pixel_threshold)
    # except ValueError as e:
        # logger.warning(f'Falling back to rebuild path for {input_pickle}: {e}')
    snr = _rebuild_snr(pipeline, snr_config, input_pickle, system_name, band, exposure)

    return (f'{system_name}_{band}', snr)


def _rebuild_snr(pipeline, snr_config, input_pickle, system_name, band, original_exposure):
    # locate the StrongLens pickle (step 02 — no substructure attached) and the
    # SyntheticImage pickle (step 04 — used only for instrument_params so the
    # rebuilt PSF matches the original).
    if pipeline.instrument_name == 'roman':
        sca_dir = os.path.basename(os.path.dirname(input_pickle))
        lens_pickle_path = os.path.join(pipeline.pipeline_dir, '02', sca_dir, f'lens_{system_name}.pkl')
        synth_dir = os.path.join(pipeline.step_dir('04'), sca_dir)
    else:
        lens_pickle_path = os.path.join(pipeline.pipeline_dir, '02', f'lens_{system_name}.pkl')
        synth_dir = pipeline.step_dir('04')

    # step 04 may have written either a full pickle or a lightweight .npz
    synth_pickle_path = os.path.join(synth_dir, f'SyntheticImage_{system_name}_{band}.pkl')
    if not os.path.exists(synth_pickle_path):
        synth_pickle_path = os.path.join(synth_dir, f'SyntheticImage_{system_name}_{band}.npz')

    try:
        lens = util.unpickle(lens_pickle_path)
        old_synthetic_image = util.load_synthetic_image(synth_pickle_path)
        instrument_params = old_synthetic_image.instrument_params

        supersampling_factor = snr_config['snr_supersampling_factor']
        kwargs_numerics = {
            'supersampling_factor': supersampling_factor,
            'compute_mode': snr_config['snr_supersampling_compute_mode'],
        }

        psf_config = pipeline.config['psf']
        num_pix = psf_config['num_pixes'][0]
        get_psf_args = {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
            'require_cached': True,
        }
        if pipeline.instrument_name == 'roman':
            get_psf_args['detector'] = instrument_params['detector']
            get_psf_args['detector_position'] = instrument_params['detector_position']
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

        synthetic_image = SyntheticImage(
            strong_lens=lens,
            instrument=pipeline.instrument,
            band=band,
            fov_arcsec=snr_config['snr_fov_arcsec'],
            instrument_params=instrument_params,
            kwargs_numerics=kwargs_numerics,
            kwargs_psf=kwargs_psf,
            pieces=True,
        )

        exposure = Exposure(
            synthetic_image,
            exposure_time=snr_config['snr_exposure_time'],
            engine=pipeline.config['imaging']['engine'],
            # engine_params=original_exposure.engine_params,
            engine_params=pipeline.config['imaging']['engine_params']
        )

        return exposure.get_snr(snr_per_pixel_threshold=snr_config['snr_per_pixel_threshold'])
    except Exception as e:
        logger.error(f'Rebuild failed for {input_pickle}: {e}')
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate signal-to-noise ratios.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False, help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
