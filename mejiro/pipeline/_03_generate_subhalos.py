import argparse
import os
import shutil
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pyHalo.preset_models import preset_model_from_name
from tqdm import tqdm

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

    # retrieve configuration parameters
    use_jax = config['jaxtronomy']['use_jax']
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

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        if pipeline.verbose: print(f'Limiting to {pipeline.limit} lens(es)')
        input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        if pipeline.limit < count:
            count = pipeline.limit
    if pipeline.verbose: print(f'Processing {count} lens(es)')

    # add subhalos to a subset of systems
    if subhalo_config['fraction'] < 1.0:
        if pipeline.verbose: print(f'Adding subhalos to {subhalo_config["fraction"] * 100}% of the systems')
        np.random.seed(config['seed'])
        mask = np.zeros(count, dtype=bool)
        num_true = int(np.round(subhalo_config['fraction'] * count))
        mask[:num_true] = True
        np.random.shuffle(mask)
    else:
        mask = np.ones(count, dtype=bool)

    # tuple the parameters
    tuple_list = [(pipeline, subhalo_config, use_jax, input_pickle, add_subhalos) for input_pickle, add_subhalos in zip(input_pickles, mask)]

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
    (pipeline, subhalo_config, use_jax, input_pickle, add_subhalos) = tuple

    if add_subhalos:
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
            print(f'Failed to generate subhalos for {lens.name}: {e}. Pickling to {failed_pickle_path}')
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
    args = parser.parse_args()
    main(args)
