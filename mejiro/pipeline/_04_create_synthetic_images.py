"""
Generates synthetic images: idealized images with no noise or detector effects (optionally, convolved with PSF).

This script creates synthetic images for each lensing system identified in previous pipeline steps, using instrument-specific parameters and PSF models. It reads a YAML configuration file specifying survey, instrument, and image simulation options. Multiprocessing is used to parallelize image generation across available CPU cores.

Usage:
    python3 _04_create_synthetic_images.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import os

# Pin BLAS/OpenMP to a single thread per worker before numpy is imported. Each
# ProcessPoolExecutor worker otherwise spawns its own BLAS thread pool, which
# oversubscribes the cores when many workers run concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from tqdm import tqdm

from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '03'
SCRIPT_NAME = '04'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']


def _is_deflector_only(name, seed, fraction):
    """Deterministically decide whether a system is rendered deflector-only.

    Keyed on ``seed`` + lens ``name`` (not on list order), so the same system is
    galaxy-only in every band and the choice is stable across ``--resume`` and
    the size-sort/limit reordering step 04 does. The realized fraction is
    approximate (binomial); see ``_03_generate_subhalos.py`` for an exact-count
    index-mask alternative if that is ever required.
    """
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    h = hashlib.md5(f'{seed}_deflector_only_{name}'.encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000 < fraction


def main(args):
    start = time.time()

    # initialize PipelineHelper; preserve any existing outputs so an interrupted
    # run can be resumed (per-(lens, band) skip logic below handles dedup).
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
    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']

    # 'full' pickles the entire SyntheticImage (current default); 'lightweight'
    # writes a compact .npz used by the romanisim path only.
    serialization = synthetic_image_config['serialization']
    if serialization not in ('full', 'lightweight'):
        raise ValueError(
            f"synthetic_image.serialization must be 'full' or 'lightweight', got {serialization!r}"
        )
    output_ext = '.npz' if serialization == 'lightweight' else '.pkl'

    # set up jaxstronomy
    if pipeline.config['jaxtronomy']['use_jax']:
        os.environ['JAX_PLATFORM_NAME'] = pipeline.config['jaxtronomy'].get('jax_platform', 'cpu')

    # set input and output directories
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='lens', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name == 'hwo' or pipeline.instrument_name == 'jwst':
        input_pickles = pipeline.retrieve_pickles(prefix='lens', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # skip lenses whose outputs all already exist (resume an interrupted run).
    bands = synthetic_image_config['bands']

    def _output_dir_for(input_pickle):
        if pipeline.instrument_name == 'roman':
            sca = pipeline.parse_sca_from_filename(input_pickle)
            return os.path.join(pipeline.output_dir, roman_util.get_sca_string(sca).lower())
        return pipeline.output_dir

    def _lens_name(input_pickle):
        base = os.path.basename(input_pickle)
        assert base.startswith('lens_') and base.endswith('.pkl'), base
        return base[len('lens_'):-len('.pkl')]

    def _is_complete(input_pickle):
        # A lens counts as complete only if every band has a real output file.
        # failed_*.pkl is NOT treated as done -- failed bands are retried.
        out_dir = _output_dir_for(input_pickle)
        name = _lens_name(input_pickle)
        for band in bands:
            if not os.path.exists(os.path.join(out_dir, f'SyntheticImage_{name}_{band}{output_ext}')):
                return False
        return True

    if args.resume:
        total_before = len(input_pickles)
        input_pickles = [p for p in input_pickles if not _is_complete(p)]
        skipped = total_before - len(input_pickles)
        logger.info(
            f'Resuming: {skipped} of {total_before} lens(es) already complete, '
            f'{len(input_pickles)} remaining.'
        )

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} lens(es)')

    # Submit substructured systems first to flatten the long tail. Substructured
    # lens pickles embed the pyhalo realization and are much larger on disk than
    # non-substructured ones, so descending file size is a cheap, robust proxy
    # that avoids unpickling every lens up front.
    if not args.sequential:
        input_pickles.sort(key=os.path.getsize, reverse=True)

    # tuple the parameters
    seed = pipeline.config['seed']
    deflector_only_fraction = synthetic_image_config.get('deflector_only_fraction', 0.0)
    if deflector_only_fraction:
        logger.info(
            f'Rendering ~{deflector_only_fraction * 100:.0f}% of systems as deflector-only '
            f'(lens galaxy light only, no source or lensing)'
        )
    tuple_list = [(pipeline, synthetic_image_config, psf_config, input_pickle, serialization,
                   _is_deflector_only(_lens_name(input_pickle), seed, deflector_only_fraction))
                  for input_pickle in input_pickles]

    # submit tasks to the executor
    try:
        with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
            futures = {executor.submit(create_synthetic_image, task): task for task in tuple_list}

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # Get the result to propagate exceptions if any
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def create_synthetic_image(input):
    # unpack tuple
    (pipeline, synthetic_image_config, psf_config, input_pickle, serialization, deflector_only) = input
    output_ext = '.npz' if serialization == 'lightweight' else '.pkl'

    # unpack pipeline params
    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    pieces = synthetic_image_config['pieces']
    num_pix = psf_config['num_pixes'][0]
    divide_up_detector = psf_config.get('divide_up_detector')  # Roman-specific parameter

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

        sca_string = roman_util.get_sca_string(instrument_params['detector']).lower()
        output_dir = os.path.join(pipeline.output_dir, sca_string)
    else:
        output_dir = pipeline.output_dir

    # generate synthetic images
    for band in bands:
        # skip if real output already exists; failed_*.pkl from a prior run is NOT skipped, the band is retried
        output_path = os.path.join(output_dir, f'SyntheticImage_{lens.name}_{band}{output_ext}')
        failed_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
        if os.path.exists(output_path):
            continue
        if os.path.exists(failed_path):
            try:
                os.remove(failed_path)
            except OSError:
                pass

        # get PSF
        get_psf_args |= {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
            'require_cached': True,
        }
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

        # build and serialize synthetic image
        try:
            synthetic_image = SyntheticImage(strong_lens=lens,
                                        instrument=pipeline.instrument,
                                        band=band,
                                        fov_arcsec=fov_arcsec,
                                        instrument_params=instrument_params,
                                        kwargs_numerics=kwargs_numerics,
                                        kwargs_psf=kwargs_psf,
                                        pieces=pieces,
                                        deflector_only=deflector_only)
            if serialization == 'lightweight':
                synthetic_image.save_lightweight(output_path)
            else:
                util.pickle(output_path, synthetic_image)
        except Exception as e:
            util.pickle(failed_path, lens)
            logger.warning(f'Error creating synthetic image for lens {lens.name} in band {band}: {e}. Pickling to {failed_path}')
            return

        # sidecar so step 05 can group images by PSF bucket without reloading pickles
        if pipeline.instrument_name == 'roman':
            sidecar_path = output_path + '.psfpos.json'
            tmp_path = sidecar_path + '.tmp'
            x, y = instrument_params['detector_position']
            with open(tmp_path, 'w') as f:
                json.dump({
                    'detector_position': [int(x), int(y)],
                    'divide_up_detector': int(divide_up_detector),
                }, f)
            os.replace(tmp_path, sidecar_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic images")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False, help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
