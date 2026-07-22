"""
Generates dark matter subhalo realizations for strong lensing systems.

This script processes previously detected lensing systems, adding dark matter substructure realizations generated using the pyHalo package to each system according to parameters specified in a mejiro configuration YAML file. It supports multiple instruments (Roman, HWO), and outputs updated lens objects and subhalo realizations for downstream analysis.

Usage:
    python3 _03_generate_subhalos.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from pyHalo.preset_models import preset_model_from_name
from tqdm import tqdm

from mejiro.analysis import lensing
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '03'
SUPPORTED_INSTRUMENTS = ['roman', 'hwo']


def main(args):
    start = time.time()

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
    use_jax = pipeline.config['jaxtronomy']['use_jax']
    subhalo_config = pipeline.config['subhalos']

    # 'full' pickles the whole realization object; 'lightweight' strips it (~76%
    # of a subhalo'd pickle) since step 04 ray-shoots from the baked-in
    # kwargs_lens, not the realization. Mirrors synthetic_image.serialization.
    serialization = subhalo_config['serialization']
    if serialization not in ('full', 'lightweight'):
        raise ValueError(
            f"subhalos.serialization must be 'full' or 'lightweight', got {serialization!r}"
        )

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
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} lens(es)')

    # add subhalos to a subset of systems
    if subhalo_config['fraction'] < 1.0:
        logger.info(f'Adding subhalos to {subhalo_config["fraction"] * 100}% of the systems')
        np.random.seed(pipeline.config['seed'])
        mask = np.zeros(count, dtype=bool)
        num_true = int(np.round(subhalo_config['fraction'] * count))
        mask[:num_true] = True
        np.random.shuffle(mask)
    else:
        mask = np.ones(count, dtype=bool)

    # tuple the parameters
    tuple_list = [(pipeline, subhalo_config, use_jax, input_pickle, add_subhalos) for input_pickle, add_subhalos in zip(input_pickles, mask)]

    # resume: skip systems whose output pickle already exists. failed_*.pkl is NOT treated as done -- it will be retried.
    total = len(tuple_list)
    if args.resume:
        filtered_tuple_list = [t for t in tuple_list if not os.path.exists(_output_target(pipeline, t[3]))]
        skipped = total - len(filtered_tuple_list)
        logger.info(
            f'Resuming: {skipped} of {total} lens(es) already complete, '
            f'{len(filtered_tuple_list)} remaining.'
        )
    else:
        filtered_tuple_list = tuple_list

    if not filtered_tuple_list:
        logger.info('All systems already processed. Nothing to do.')
        stop = time.time()
        execution_time = util.print_execution_time(start, stop, return_string=True)
        util.write_execution_time(execution_time, SCRIPT_NAME,
                                  os.path.join(pipeline.pipeline_dir, 'execution_times.json'))
        return

    # submit tasks to the executor
    try:
        with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(len(filtered_tuple_list))) as executor:
            futures = {executor.submit(add, task): task for task in filtered_tuple_list}

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # get the result to propagate exceptions if any
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def _output_target(pipeline, input_pickle):
    """Compute the per-lens output pickle path that `add()` will write to."""
    base = os.path.basename(input_pickle)
    if pipeline.instrument_name == 'roman':
        sca_string = os.path.dirname(input_pickle).split('/')[-1]
        return os.path.join(pipeline.output_dir, sca_string, base)
    return os.path.join(pipeline.output_dir, base)


def add(tuple):
    np.random.seed()

    # unpack tuple
    (pipeline, subhalo_config, use_jax, input_pickle, add_subhalos) = tuple

    # remove any stale failure sentinel from a prior attempt so failed work is fully retried
    lens_name_for_failed = os.path.basename(input_pickle).replace('lens_', '').replace('.pkl', '')
    stale_failed = os.path.join(pipeline.output_dir, f'failed_{lens_name_for_failed}.pkl')
    if os.path.exists(stale_failed):
        try:
            os.remove(stale_failed)
        except OSError:
            pass

    if add_subhalos:
        # load the lens
        lens = util.unpickle(input_pickle)

        # set defaults for the realization_kwargs
        realization_kwargs = subhalo_config["realization_kwargs"]
        if "cone_opening_angle_arcsec" not in realization_kwargs:
            realization_kwargs["cone_opening_angle_arcsec"] = lens.get_einstein_radius() * 3

        # generate realization
        REALIZATION = preset_model_from_name(subhalo_config["pyhalo_model"])
        z_lens_rounded = round(lens.z_lens, 2)
        z_source_rounded = round(lens.z_source, 2)
        # circumvent bug with pyhalo, sometimes fails when redshifts have more than 2 decimal places.
        # Also: at certain rounded z_lens values, pyHalo's generate_lens_plane_redshifts produces a
        # duplicate plane at z_lens (FP issue with np.arange over 0.02 steps), giving delta_z=0 in
        # the two-halo term and a "R = 0.00e+00 is too small" error from colossus. Retry once with
        # z_lens shifted by +0.01 (~10 Mpc at z~2.4, well within our 2dp rounding) to avoid it.
        realization = None
        for z_lens_try in (z_lens_rounded, z_lens_rounded + 0.01):
            try:
                realization = REALIZATION(z_lens=z_lens_try,
                                    z_source=z_source_rounded,
                                    log_m_host=np.log10(lens.get_main_halo_mass()),
                                    kwargs_cosmo=util.get_kwargs_cosmo(lens.cosmo),
                                    **realization_kwargs)
                if z_lens_try != z_lens_rounded:
                    logger.warning(f'[{lens.name}] Realization succeeded after nudging z_lens '
                                   f'{z_lens_rounded} -> {z_lens_try} (pyHalo duplicate-plane workaround)')
                break
            except Exception as e:
                last_exc = e
                continue
        if realization is None:
            failed_pickle_path = os.path.join(pipeline.output_dir, f'failed_{lens.name}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(f'Failed to generate subhalos for {lens.name}: {last_exc}. Pickling to {failed_pickle_path}')
            return

        # add subhalos
        lens.add_realization(realization, use_jax=use_jax)

        # For lightweight serialization, drop the ~59 MB realization object
        # (step 04 reads only the baked-in kwargs_lens); the substructure flag is
        # recorded on the lens and a truthy sentinel preserves has_realization.
        if subhalo_config['serialization'] == 'lightweight':
            lensing.strip_realization(lens)

        # pickle the subhalo realization
        # subhalo_dir = os.path.join(pipeline.output_dir, 'subhalos')
        # util.create_directory_if_not_exists(subhalo_dir)
        # util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.name}.pkl'), realization)

        # pickle the lens with subhalos
        if pipeline.instrument_name == 'roman':
            sca_string = os.path.dirname(input_pickle).split('/')[-1]
            pipeline.output_dir = os.path.join(pipeline.output_dir, sca_string)
        pickle_target = os.path.join(pipeline.output_dir, f'lens_{lens.name}.pkl')
        util.pickle(pickle_target, lens)
    else:
        # if not adding subhalos, just copy the input pickle to the output directory
        if pipeline.instrument_name == 'roman':
            sca_string = os.path.dirname(input_pickle).split('/')[-1]
            pipeline.output_dir = os.path.join(pipeline.output_dir, sca_string)
        target = os.path.join(pipeline.output_dir, os.path.basename(input_pickle))
        shutil.copy2(input_pickle, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dark matter substructure realizations")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False, help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
