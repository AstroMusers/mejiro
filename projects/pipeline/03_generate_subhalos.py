import argparse
import multiprocessing
import os
import time
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pyHalo.PresetModels.cdm import CDM
from tqdm import tqdm

from mejiro.cosmo import cosmo
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '03'
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
    subhalo_config = config['subhalos']

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

    # add subhalos to a subset of systems
    if subhalo_config['fraction'] < 1.0:
        if pipeline.verbose: print(f'Adding subhalos to {subhalo_config["fraction"] * 100}% of the systems')
        np.random.seed(config['seed'])
        np.random.shuffle(input_pickles)
        count = int(len(input_pickles) * subhalo_config['fraction'])
        input_pickles = input_pickles[:count]

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        if pipeline.verbose: print(f'Limiting to {pipeline.limit} lens(es)')
        input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        if pipeline.limit < count:
            count = pipeline.limit
    if pipeline.verbose: print(f'Processing {count} lens(es)')

    # tuple the parameters
    tuple_list = [(pipeline, subhalo_config, input_pickle) for input_pickle in input_pickles]

    # define the number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count
    process_count -= config['headroom_cores']['script_03']
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # submit tasks to the executor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
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
    (pipeline, subhalo_config, input_pickle) = tuple

    # load the lens
    lens = util.unpickle(input_pickle)

    # convert stellar mass to main halo mass
    main_halo_mass = cosmo.stellar_to_main_halo_mass(lens.physical_params['lens_stellar_mass'], lens.z_lens, sample=True)

    try:
        cdm_realization = CDM(z_lens=round(lens.z_lens, 2),  # circumvent bug with pyhalo, sometimes fails when redshifts have more than 2 decimal places
                              z_source=round(lens.z_source, 2),
                              sigma_sub=subhalo_config['sigma_sub'],
                              log_mlow=subhalo_config['log_mlow'],
                              log_mhigh=subhalo_config['log_mhigh'],
                              log_m_host=np.log10(main_halo_mass),
                              r_tidal=subhalo_config['r_tidal'],
                              cone_opening_angle_arcsec=lens.get_einstein_radius() * 3,
                              LOS_normalization=subhalo_config['los_normalization'],
                              kwargs_cosmo=util.get_kwargs_cosmo(lens.cosmo))
    except Exception as e:
        failed_pickle_path = os.path.join(pipeline.output_dir, f'failed_{lens.name}.pkl')
        util.pickle(failed_pickle_path, lens)
        print(f'Failed to generate subhalos for {lens.name}: {e}. Pickling to {failed_pickle_path}')
        return

    # add subhalos
    lens.add_realization(cdm_realization, use_jax=False)

    # pickle the subhalo realization
    subhalo_dir = os.path.join(pipeline.output_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.name}.pkl'), cdm_realization)

    # pickle the lens with subhalos
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
