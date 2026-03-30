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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pyHalo.preset_models import preset_model_from_name
from tqdm import tqdm

from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '03'
SUPPORTED_INSTRUMENTS = ['roman']


def main(args):
    start = time.time()

    # initialize PipeLineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # retrieve configuration parameters
    use_jax = pipeline.config['jaxtronomy']['use_jax']
    subhalo_config = pipeline.config['subhalos']

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

    # tuple the parameters
    tuple_list = [(pipeline, subhalo_config, use_jax, input_pickle, rt) for input_pickle, rt in zip(input_pickles, realization_types)]

    # submit tasks to the executor
    with ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(count)) as executor:
        futures = {executor.submit(add, task): task for task in tuple_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # get the result to propagate exceptions if any

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def add(tuple):
    np.random.seed()

    # unpack tuple
    (pipeline, subhalo_config, use_jax, input_pickle, realization_type) = tuple

    if realization_type == 'CDM':
        # load the lens
        lens = util.unpickle(input_pickle)

        # set defaults for the realization_kwargs
        realization_kwargs = subhalo_config["realization_kwargs"]
        if "cone_opening_angle_arcsec" not in realization_kwargs:
            realization_kwargs["cone_opening_angle_arcsec"] = lens.get_einstein_radius() * 3

        # generate realization
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

        # add subhalos
        lens.add_realization(realization, use_jax=use_jax)

        # pickle the subhalo realization
        subhalo_dir = os.path.join(pipeline.output_dir, 'subhalos')
        util.create_directory_if_not_exists(subhalo_dir)
        util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.name}.pkl'), realization)

        # pickle the lens with subhalos
        if pipeline.instrument_name == 'roman':
            sca_string = os.path.dirname(input_pickle).split('/')[-1]
            pipeline.output_dir = os.path.join(pipeline.output_dir, sca_string)
        pickle_target = os.path.join(pipeline.output_dir, f'lens_{lens.name}.pkl')
        util.pickle(pickle_target, lens)

    elif realization_type == 'ULDM':
        # load the lens
        lens = util.unpickle(input_pickle)

        log_main_halo_mass = np.log10(lens.get_main_halo_mass())
        log_mlow = subhalo_config['realization_kwargs']['log_mlow']
        log_mhigh = subhalo_config['realization_kwargs']['log_mhigh']

        # generate ULDM realization
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

        # add realization
        lens.add_realization(realization, use_jax=use_jax)

        # pickle the subhalo realization
        subhalo_dir = os.path.join(pipeline.output_dir, 'subhalos')
        util.create_directory_if_not_exists(subhalo_dir)
        util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.name}.pkl'), realization)

        # pickle the lens with realization
        if pipeline.instrument_name == 'roman':
            sca_string = os.path.dirname(input_pickle).split('/')[-1]
            pipeline.output_dir = os.path.join(pipeline.output_dir, sca_string)
        pickle_target = os.path.join(pipeline.output_dir, f'lens_{lens.name}.pkl')
        util.pickle(pickle_target, lens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dark matter substructure realizations")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
