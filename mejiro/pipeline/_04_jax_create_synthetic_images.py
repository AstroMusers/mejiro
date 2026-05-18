"""
Generates synthetic images using JAX (via jaxtronomy) for ray-shooting.

Drop-in companion to ``_04_create_synthetic_images.py`` that routes every
JAX-supported lens model profile through jaxtronomy, verifies JAX is usable on
the current machine, and optionally groups systems with matching
``lens_model_list`` signatures to amortize JIT compilation cost on the GPU.

Outputs are written to ``<pipeline_dir>/04_jax/`` with the same filename
pattern as ``_04_create_synthetic_images.py`` so downstream steps can consume
either script's output interchangeably.

Usage:
    python3 _04_jax_create_synthetic_images.py --config <config.yaml>

Config additions (under existing top-level keys):
    cores.script_04_jax: <int>                   # process count key
    jaxtronomy.parallel_systems: <bool>          # default False
    jaxtronomy.batch_size: <int>                 # default 8

Note on "parallel" mode:
    With ``parallel_systems: True``, systems sharing an identical lens_model_list
    are grouped per band and the deflection kernel is JIT-compiled once per
    group, amortizing XLA compilation across the batch. True ``jax.vmap``
    batching of the full ``SyntheticImage`` pipeline is not yet wired up
    because ``ImageModel.image()`` mixes JAX ray-shooting with numpy-side
    supersampling/PSF convolution; extending this is left as future work.
"""
import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '03'
SCRIPT_NAME = '04_jax'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']


def _check_jax_available(jax_platform):
    """Import JAX + jaxtronomy and verify the requested platform has devices."""
    try:
        import jax
    except ImportError as e:
        raise RuntimeError(
            f"JAX is required for {SCRIPT_NAME} but failed to import: {e}"
        ) from e

    try:
        devices = jax.devices(jax_platform)
    except RuntimeError as e:
        raise RuntimeError(
            f"Requested jax_platform '{jax_platform}' is not available on this "
            f"machine: {e}"
        ) from e
    if not devices:
        raise RuntimeError(
            f"No JAX devices found for platform '{jax_platform}'."
        )
    logger.info(f"JAX platform '{jax_platform}' ready: {devices}")

    try:
        from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS
    except ImportError as e:
        raise RuntimeError(
            f"jaxtronomy is required for {SCRIPT_NAME} but failed to import: {e}"
        ) from e

    return jax, _JAXXED_MODELS


def enable_jax_on_lens(lens, jaxxed_models):
    """Flip ``lens.use_jax[i]`` to True wherever ``lens_model_list[i]`` is JAX-supported."""
    lens_model_list = lens.lens_model_list
    use_jax = list(lens.use_jax) if lens.use_jax is not None else [False] * len(lens_model_list)
    if len(use_jax) != len(lens_model_list):
        use_jax = [False] * len(lens_model_list)
    lens.use_jax = [
        True if lens_model_list[i] in jaxxed_models else bool(use_jax[i])
        for i in range(len(lens_model_list))
    ]


def _resolve_output_dir(pipeline, input_pickle):
    if pipeline.instrument_name == 'roman':
        sca_string = roman_util.get_sca_string(pipeline.parse_sca_from_filename(input_pickle)).lower()
        return os.path.join(pipeline.output_dir, sca_string)
    return pipeline.output_dir


def _build_instrument_params(pipeline, input_pickle, divide_up_detector):
    instrument_params = {}
    get_psf_args = {}
    if pipeline.instrument_name == 'roman':
        possible_detector_positions = roman_util.divide_up_sca(divide_up_detector)
        detector_position = random.choice(possible_detector_positions)
        instrument_params['detector'] = pipeline.parse_sca_from_filename(input_pickle)
        instrument_params['detector_position'] = detector_position
        get_psf_args |= instrument_params
    return instrument_params, get_psf_args


def create_synthetic_image(pipeline, synthetic_image_config, psf_config, input_pickle, jaxxed_models):
    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    pieces = synthetic_image_config['pieces']
    num_pix = psf_config['num_pixes'][0]
    divide_up_detector = psf_config.get('divide_up_detector')

    lens = util.unpickle(input_pickle)
    enable_jax_on_lens(lens, jaxxed_models)

    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode,
    }

    instrument_params, get_psf_args = _build_instrument_params(pipeline, input_pickle, divide_up_detector)
    output_dir = _resolve_output_dir(pipeline, input_pickle)

    for band in bands:
        get_psf_args |= {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
            'require_cached': True,
        }
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

        try:
            synthetic_image = SyntheticImage(
                strong_lens=lens,
                instrument=pipeline.instrument,
                band=band,
                fov_arcsec=fov_arcsec,
                instrument_params=instrument_params,
                kwargs_numerics=kwargs_numerics,
                kwargs_psf=kwargs_psf,
                pieces=pieces,
            )
            util.pickle(os.path.join(output_dir, f'SyntheticImage_{lens.name}_{band}.pkl'), synthetic_image)
        except Exception as e:
            failed_pickle_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(
                f'Error creating synthetic image for lens {lens.name} in band {band}: {e}. '
                f'Pickling to {failed_pickle_path}'
            )
            return


def _run_sequential(pipeline, synthetic_image_config, psf_config, input_pickles, jaxxed_models):
    for input_pickle in tqdm(input_pickles):
        create_synthetic_image(pipeline, synthetic_image_config, psf_config, input_pickle, jaxxed_models)


def _run_bucketed(pipeline, synthetic_image_config, psf_config, input_pickles, jaxxed_models, batch_size):
    """Group input pickles by lens_model_list signature to amortize JIT warmup.

    Pickles are bucketed on their on-disk ``lens.lens_model_list`` tuple. Within
    each bucket the first system triggers JIT compilation of the jaxtronomy
    deflection kernel; subsequent systems in the bucket reuse the cached
    compilation. ``batch_size`` caps bucket size so that very large groups are
    processed in chunks (useful as a memory ceiling on GPU).
    """
    signature_to_pickles = defaultdict(list)
    for input_pickle in tqdm(input_pickles, desc='Bucketing by lens_model_list'):
        lens = util.unpickle(input_pickle)
        signature = tuple(lens.lens_model_list)
        signature_to_pickles[signature].append(input_pickle)

    logger.info(
        f'Bucketed {len(input_pickles)} systems into {len(signature_to_pickles)} '
        f'lens_model_list signature group(s)'
    )

    with tqdm(total=len(input_pickles)) as pbar:
        for signature, pickles in signature_to_pickles.items():
            for start in range(0, len(pickles), batch_size):
                chunk = pickles[start:start + batch_size]
                for input_pickle in chunk:
                    create_synthetic_image(
                        pipeline, synthetic_image_config, psf_config, input_pickle, jaxxed_models,
                    )
                    pbar.update(1)


def main(args):
    start = time.time()

    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']
    jax_config = pipeline.config.get('jaxtronomy', {})

    jax_platform = jax_config.get('jax_platform', 'cpu')
    parallel_systems = bool(jax_config.get('parallel_systems', False))
    batch_size = int(jax_config.get('batch_size', 8))

    os.environ['JAX_PLATFORM_NAME'] = jax_platform
    _jax, jaxxed_models = _check_jax_available(jax_platform)
    if parallel_systems and jax_platform == 'cpu':
        logger.warning(
            'parallel_systems=True with jax_platform=cpu: JIT warmup amortization '
            'offers only modest gains on CPU.'
        )

    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='lens', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name in ('hwo', 'jwst'):
        input_pickles = pipeline.retrieve_pickles(prefix='lens', suffix='', extension='.pkl')
    else:
        raise ValueError(
            f'Unknown instrument {pipeline.instrument_name}. '
            f'Supported instruments are {SUPPORTED_INSTRUMENTS}.'
        )

    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} lens(es)')

    if not args.sequential:
        input_pickles.sort(key=os.path.getsize, reverse=True)

    if parallel_systems:
        _run_bucketed(
            pipeline, synthetic_image_config, psf_config, input_pickles, jaxxed_models, batch_size,
        )
    else:
        _run_sequential(
            pipeline, synthetic_image_config, psf_config, input_pickles, jaxxed_models,
        )

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(
        execution_time, SCRIPT_NAME,
        os.path.join(pipeline.pipeline_dir, 'execution_times.json'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic images using JAX/jaxtronomy")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    args = parser.parse_args()
    main(args)
