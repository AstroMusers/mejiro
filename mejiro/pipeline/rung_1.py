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
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pyHalo.preset_models import preset_model_from_name
from tqdm import tqdm

from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '05'
SUPPORTED_INSTRUMENTS = ['roman']
REALIZATION_TYPE = 'CDM'  # set to 'CDM' or 'ULDM' to select which systems to process


def main(args):
    start = time.time()

    # initialize PipeLineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # retrieve configuration parameters
    use_jax = pipeline.config['jaxtronomy']['use_jax']
    subhalo_config = pipeline.config['subhalos']
    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']
    imaging_config = pipeline.config['imaging']

    # set up jaxstronomy
    if use_jax:
        os.environ['JAX_PLATFORM_NAME'] = pipeline.config['jaxtronomy'].get('jax_platform', 'cpu')

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
        input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        if pipeline.limit < count:
            count = pipeline.limit
    logger.info(f'Processing {count} lens(es)')

    # assign half the systems CDM realizations, half ULDM, randomly
    np.random.seed(pipeline.config['seed'])
    half = count // 2
    realization_types = ['CDM'] * half + ['ULDM'] * (count - half)
    np.random.shuffle(realization_types)
    logger.info(f'Assigning {half} system(s) CDM realizations and {count - half} system(s) ULDM realizations')

    # save full assignments for future runs
    assignments = {pickle_path: rt for pickle_path, rt in zip(input_pickles, realization_types)}
    assignments_path = os.path.join(pipeline.output_dir, 'realization_assignments.pkl')
    util.pickle(assignments_path, assignments)
    logger.info(f'Saved realization assignments to {assignments_path}')

    # filter to the selected realization type
    selected_pickles = [p for p, rt in zip(input_pickles, realization_types) if rt == REALIZATION_TYPE]
    selected_count = len(selected_pickles)
    logger.info(f'Processing {selected_count} {REALIZATION_TYPE} system(s), deferring {count - selected_count} other system(s)')

    # tuple the parameters
    tuple_list = [(pipeline, subhalo_config, use_jax, input_pickle, REALIZATION_TYPE,
                   synthetic_image_config, psf_config, imaging_config)
                  for input_pickle in selected_pickles]

    # submit tasks to the executor
    try:
        with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(selected_count)) as executor:
            futures = {executor.submit(add, task): task for task in tuple_list}

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


def add(tuple):
    np.random.seed()

    # unpack tuple
    (pipeline, subhalo_config, use_jax, input_pickle, realization_type,
     synthetic_image_config, psf_config, imaging_config) = tuple

    # load the lens
    t0 = time.time()
    lens = util.unpickle(input_pickle)
    logger.info(f'[{os.path.basename(input_pickle)}] Unpickled lens in {time.time() - t0:.2f}s')

    # --- Phase 1: Generate and add realization ---
    if realization_type == 'CDM':
        # set defaults for the realization_kwargs
        realization_kwargs = subhalo_config["realization_kwargs"]
        if "cone_opening_angle_arcsec" not in realization_kwargs:
            realization_kwargs["cone_opening_angle_arcsec"] = lens.get_einstein_radius() * 3

        # generate realization
        t1 = time.time()
        try:
            REALIZATION = preset_model_from_name(subhalo_config["pyhalo_model"])
            realization = REALIZATION(z_lens=round(lens.z_lens, 2),  # circumvent bug with pyhalo, sometimes fails when redshifts have more than 2 decimal places
                                z_source=round(lens.z_source, 2),
                                log_m_host=np.log10(lens.get_main_halo_mass()),
                                kwargs_cosmo=util.get_kwargs_cosmo(lens.cosmo),
                                **realization_kwargs)
        except Exception as e:
            failed_pickle_path = os.path.join(pipeline.output_dir, f'failed_{lens.name}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(f'Failed to generate CDM subhalos for {lens.name}: {e}. Pickling to {failed_pickle_path}')
            return
        logger.info(f'[{lens.name}] Generated CDM realization in {time.time() - t1:.2f}s')

        # add subhalos
        t1 = time.time()
        lens.add_realization(realization, use_jax=use_jax)
        logger.info(f'[{lens.name}] Added CDM realization in {time.time() - t1:.2f}s')

    elif realization_type == 'ULDM':
        log_main_halo_mass = np.log10(lens.get_main_halo_mass())
        log_mlow = subhalo_config['realization_kwargs']['log_mlow']
        log_mhigh = subhalo_config['realization_kwargs']['log_mhigh']

        # generate ULDM realization
        t1 = time.time()
        try:
            ULDM = preset_model_from_name('ULDM')
            realization = ULDM(round(lens.z_lens, 2),
                               round(lens.z_source, 2),
                               log10_m_uldm=-21,
                               cone_opening_angle_arcsec=5,
                               log_m_host=log_main_halo_mass,
                               flucs_shape='ring',
                               flucs_args={'angle': 0.0, 'rmin': 0.9, 'rmax': 1.1},
                               log10_fluc_amplitude=-1.6,
                               n_cut=1000000,
                               log_mlow=log_mlow,
                               log_mhigh=log_mhigh)
        except Exception as e:
            failed_pickle_path = os.path.join(pipeline.output_dir, f'failed_{lens.name}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(f'Failed to generate ULDM realization for {lens.name}: {e}. Pickling to {failed_pickle_path}')
            return
        logger.info(f'[{lens.name}] Generated ULDM realization in {time.time() - t1:.2f}s')

        # add realization
        t1 = time.time()
        lens.add_realization(realization, use_jax=use_jax)
        logger.info(f'[{lens.name}] Added ULDM realization in {time.time() - t1:.2f}s')

    # --- Phase 2: Synthetic image + exposure generation ---

    # extract synthetic image config
    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    pieces = synthetic_image_config['pieces']
    num_pix = psf_config['num_pixes'][0]
    divide_up_detector = psf_config.get('divide_up_detector')

    # extract imaging config
    exposure_time = imaging_config['exposure_time']
    engine = imaging_config['engine']
    engine_params = imaging_config['engine_params']

    # build kwargs_numerics
    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode
    }

    get_psf_args = {}
    instrument_params = {}

    # set detector and pick random position (Roman-specific)
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

    # generate synthetic images and exposures for each band
    for band in bands:
        # get PSF kwargs
        get_psf_args_band = get_psf_args.copy()
        get_psf_args_band |= {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
        }
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args_band)

        # create SyntheticImage
        t_band = time.time()
        try:
            synthetic_image = SyntheticImage(
                strong_lens=lens,
                instrument=pipeline.instrument,
                band=band,
                fov_arcsec=fov_arcsec,
                instrument_params=instrument_params,
                kwargs_numerics=kwargs_numerics,
                kwargs_psf=kwargs_psf,
                pieces=pieces)
        except Exception as e:
            failed_pickle_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(f'Error creating synthetic image for lens {lens.name} in band {band}: {e}. Pickling to {failed_pickle_path}')
            return
        logger.info(f'[{lens.name}] Created SyntheticImage for {band} in {time.time() - t_band:.2f}s')

        # create Exposure from SyntheticImage
        t_exp = time.time()
        try:
            exposure = Exposure(
                synthetic_image,
                exposure_time=exposure_time,
                engine=engine,
                engine_params=engine_params)
        except Exception as e:
            failed_pickle_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
            util.pickle(failed_pickle_path, lens)
            logger.warning(f'Error creating exposure for lens {lens.name} in band {band}: {e}. Pickling to {failed_pickle_path}')
            return
        logger.info(f'[{lens.name}] Created Exposure for {band} in {time.time() - t_exp:.2f}s')

        # save exposure data as .npy
        np.save(os.path.join(output_dir, f'exposure_{lens.name}_{band}.npy'), exposure.data)

    # pickle the strong lens (once, not per band)
    util.pickle(os.path.join(output_dir, f'lens_{lens.name}.pkl'), lens)

    logger.info(f'[{lens.name}] Total processing time: {time.time() - t0:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dark matter substructure realizations")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
