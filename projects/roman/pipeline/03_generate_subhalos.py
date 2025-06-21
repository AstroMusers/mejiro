import argparse
import multiprocessing
import os
import sys
import time
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pyHalo.PresetModels.cdm import CDM
from tqdm import tqdm


PREV_SCRIPT_NAME = '02'
SCRIPT_NAME = '03'


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
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    limit = config['limit']
    scas = config['survey']['scas']
    subhalo_config = config['subhalos']

    # set up top directory for all pipeline output
    pipeline_dir = os.path.join(data_dir, config['pipeline_dir'])
    if dev:
        pipeline_dir += '_dev'

    # tell script where the output of previous script is
    input_dir = os.path.join(pipeline_dir, PREV_SCRIPT_NAME)
    input_sca_dirs = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in input_sca_dirs])
    scas = [str(sca).zfill(2) for sca in scas]
    if verbose: print(f'Reading from {input_sca_dirs}')

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    output_sca_dirs = []
    for sca in scas:
        sca_dir = os.path.join(output_dir, f'sca{sca}')
        os.makedirs(sca_dir, exist_ok=True)
        output_sca_dirs.append(sca_dir)
    if verbose: print(f'Set up output directories {output_sca_dirs}')

    # parse uids
    uid_dict = {}
    for sca in scas:
        pickled_lenses = sorted(glob(input_dir + f'/sca{sca}/lens_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[1].split('.')[0] for i in pickled_lenses]
        uid_dict[sca] = lens_uids
    count = 0
    for sca, lens_uids in uid_dict.items():
        count += len(lens_uids)
    if limit is not None:
        if verbose: print(f'Limiting to {limit} lens(es)')
        lens_uids = lens_uids[:limit]
        if limit < count:
            count = limit
    if verbose: print(f'Processing {count} lens(es)')

    # tuple the parameters
    tuple_list = []
    for i, (sca, lens_uids) in enumerate(uid_dict.items()):
        input_sca_dir = os.path.join(input_dir, f'sca{sca}')
        output_sca_dir = os.path.join(output_dir, f'sca{sca}')

        for uid in lens_uids:
            tuple_list.append((uid, subhalo_config, input_sca_dir, output_sca_dir))

    # add subhalos to a subset of systems
    if subhalo_config['fraction'] < 1.0:
        if verbose: print(f'Adding subhalos to {subhalo_config["fraction"] * 100}% of the systems')
        np.random.seed(config['seed'])
        np.random.shuffle(tuple_list)
        count = int(len(tuple_list) * subhalo_config['fraction'])
        tuple_list = tuple_list[:count]

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
                              os.path.join(pipeline_dir, 'execution_times.json'))


def add(tuple):
    np.random.seed()

    from mejiro.cosmo import cosmo
    from mejiro.utils import util

    # unpack tuple
    (uid, subhalo_config, input_dir, output_dir) = tuple

    # unpack pipeline_params
    los_normalization = subhalo_config['los_normalization']
    r_tidal = subhalo_config['r_tidal']
    sigma_sub = subhalo_config['sigma_sub']
    log_mlow = subhalo_config['log_mlow']
    log_mhigh = subhalo_config['log_mhigh']

    # load the lens based on uid
    lens = util.unpickle(os.path.join(input_dir, f'lens_{uid}.pkl'))
    lens_uid = lens.name.split('_')[-1].split('.')[0]
    assert lens_uid == uid, f'UID mismatch: {lens_uid} != {uid}'

    main_halo_mass = cosmo.stellar_to_main_halo_mass(lens.physical_params['lens_stellar_mass'], lens.z_lens, sample=True)
    log_m_host = np.log10(main_halo_mass)
    kwargs_cosmo = util.get_kwargs_cosmo(lens.cosmo)

    # circumvent bug with pyhalo, sometimes fails when redshifts have more than 2 decimal places
    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)

    # get Einstein radius
    einstein_radius = lens.get_einstein_radius()

    try:
        cdm_realization = CDM(z_lens,
                              z_source,
                              sigma_sub=sigma_sub,
                              log_mlow=log_mlow,
                              log_mhigh=log_mhigh,
                              log_m_host=log_m_host,
                              r_tidal=r_tidal,
                              cone_opening_angle_arcsec=einstein_radius * 3,
                              LOS_normalization=los_normalization,
                              kwargs_cosmo=kwargs_cosmo)
    except Exception as e:
        print(f'Failed to generate subhalos for lens {lens_uid}: {e}')
        return

    # cdm_realization = CDM(z_lens,
    #                           z_source,
    #                           sigma_sub=sigma_sub,
    #                           log_mlow=log_mlow,
    #                           log_mhigh=log_mhigh,
    #                           log_m_host=log_m_host,
    #                           r_tidal=r_tidal,
    #                           cone_opening_angle_arcsec=einstein_radius * 3,
    #                           LOS_normalization=los_normalization,
    #                           kwargs_cosmo=kwargs_cosmo)

    # Add subhalos
    lens.add_realization(cdm_realization)

    # Pickle the subhalo realization
    subhalo_dir = os.path.join(output_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{str(uid).zfill(8)}.pkl'), cdm_realization)

    # Pickle the lens with subhalos
    pickle_target = os.path.join(output_dir, f'lens_{str(uid).zfill(8)}.pkl')
    util.pickle(pickle_target, lens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and cache Roman PSFs.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
