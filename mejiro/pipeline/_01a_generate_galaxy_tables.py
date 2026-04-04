"""
Pre-generates galaxy population tables for the survey simulation.

This script generates a configurable number of galaxy tables by running SkyPy and
SLHammocks pipelines, then serializes them as pickle files. This separates the
expensive galaxy population generation from the survey simulation itself, allowing
the simulation step (_01b) to load pre-computed tables and skip the initialization.

Each table is generated with a unique random seed for reproducibility. Tables are
associated with detectors (for Roman, each table is assigned a detector via
round-robin), since SkyPy configs differ per detector.

Usage:
    python3 _01a_generate_galaxy_tables.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.cosmology import default_cosmology
from astropy.units import Quantity
import slsim.Pipelines as pipelines
from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline
from tqdm import tqdm

import mejiro
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = None
SCRIPT_NAME = '01a'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']


def main(args):
    start = time.time()

    # initialize PipelineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # set configuration parameters
    pipeline.config['survey']['cosmo'] = default_cosmology.get()

    num_galaxy_tables = pipeline.config['survey'].get('num_galaxy_tables', pipeline.runs)

    # build task list: one task per galaxy table
    tuple_list = []
    for table_index in range(num_galaxy_tables):
        if pipeline.instrument.num_detectors > 1:
            detector = pipeline.detectors[table_index % len(pipeline.detectors)]
        else:
            detector = None
        tuple_list.append((table_index, detector, pipeline.config, pipeline.output_dir, pipeline.instrument))

    # process the tasks with ProcessPoolExecutor
    num_workers = pipeline.calculate_process_count(num_galaxy_tables)
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_galaxy_table, task) for task in tuple_list]

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # propagate any exceptions
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    logger.info(f'Generated {num_galaxy_tables} galaxy table(s)')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '01a', os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def generate_galaxy_table(tuple):
    module_path = os.path.dirname(mejiro.__file__)

    # unpack tuple
    table_index, detector, config, output_dir, instrument = tuple

    # seed for reproducibility
    global_seed = config.get('seed', 42)
    np.random.seed(hash((global_seed, 'table', table_index)) % (2**32))

    # suppress warnings
    if config['suppress_warnings']:
        warnings.filterwarnings("ignore")

    survey_config = config['survey']
    cosmo = survey_config['cosmo']
    area = survey_config['area']
    use_slhammocks_pipeline = survey_config['use_slhammocks_pipeline']

    # determine detector string for config file lookup
    if instrument.name == 'Roman':
        detector_string = roman_util.get_sca_string(detector).lower()

    # load speclite filters (must be registered before SkyPy can use them)
    if config['pipeline_label'] == 'all':
        from mejiro.instruments.hst import HST
        from mejiro.instruments.jwst import JWST
        from mejiro.instruments.roman import Roman
        from mejiro.instruments.hwo import HWO
        Roman.load_speclite_filters(detector='SCA01')
        HST.load_speclite_filters()
        JWST.load_speclite_filters()
        HWO.load_speclite_filters()
    elif instrument.name == 'Roman':
        instrument.load_speclite_filters(detector=detector_string)
    elif instrument.name == 'HWO' or instrument.name == 'JWST':
        instrument.load_speclite_filters()

    # load SkyPy config file
    cache_dir = os.path.join(module_path, 'data', 'skypy', survey_config['skypy_config'])
    if instrument.name == 'Roman':
        skypy_config = os.path.join(cache_dir,
                                    f'{survey_config["skypy_config"]}_{detector_string}.yml')
    elif instrument.name == 'HWO' or instrument.name == 'JWST':
        skypy_config = os.path.join(cache_dir, survey_config['skypy_config'] + '.yml')
    if not os.path.exists(skypy_config):
        raise FileNotFoundError(f'SkyPy configuration file {skypy_config} not found.')
    config_file = util.load_skypy_config(skypy_config)
    logger.info(f'[Table {table_index}] Loaded SkyPy configuration file {skypy_config}')

    # load SLHammocks SkyPy config file
    slhammocks_pipeline_kwargs = None
    if use_slhammocks_pipeline:
        slhammocks_pipeline_kwargs = dict(survey_config['slhammocks_pipeline_kwargs'])
        cache_dir = os.path.join(module_path, 'data', 'skypy', 'slhammocks')
        if config['pipeline_label'] == 'all':
            slhammocks_config = os.path.join(cache_dir, 'slhammocks_all.yml')
        elif instrument.name == 'Roman':
            slhammocks_config = os.path.join(cache_dir,
                                             f'{slhammocks_pipeline_kwargs["skypy_config"]}_{detector_string}.yml')
        elif instrument.name == 'HWO' or instrument.name == 'JWST':
            slhammocks_config = os.path.join(cache_dir, slhammocks_pipeline_kwargs["skypy_config"] + '.yml')
        if not os.path.exists(slhammocks_config):
            raise FileNotFoundError(f'SLHammocks configuration file {slhammocks_config} not found.')
        slhammocks_pipeline_kwargs['skypy_config'] = slhammocks_config
        logger.info(f'[Table {table_index}] Loaded SLHammocks configuration file {slhammocks_config}')

    # validate survey area
    survey_area = float(config_file['fsky'][:-5])
    sky_area = Quantity(value=survey_area, unit='deg2')
    assert sky_area.value == area, f'Area mismatch: {sky_area.value} != {area}'

    # run SkyPy pipeline
    logger.info(f'[Table {table_index}] Generating galaxy population...')
    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=skypy_config,
        sky_area=sky_area,
        filters=None,
        cosmo=cosmo
    )

    # build the output data
    galaxy_data = {
        'red_galaxies': galaxy_simulation_pipeline.red_galaxies,
        'blue_galaxies': galaxy_simulation_pipeline.blue_galaxies,
        'detector': detector,
    }

    # run SLHammocks pipeline if enabled
    if use_slhammocks_pipeline:
        halo_galaxy_pipeline = SLHammocksPipeline(
            sky_area=sky_area,
            cosmo=cosmo,
            z_min=survey_config['deflector_z_min'],
            z_max=survey_config['deflector_z_max'],
            **slhammocks_pipeline_kwargs
        )
        galaxy_data['halo_galaxies'] = halo_galaxy_pipeline.halo_galaxies

    # serialize
    output_path = os.path.join(output_dir, f'galaxy_table_{str(table_index).zfill(4)}.pkl')
    util.pickle(output_path, galaxy_data)
    logger.info(f'[Table {table_index}] Saved galaxy table to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate galaxy population tables for survey simulation")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False, help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    args = parser.parse_args()
    main(args)
