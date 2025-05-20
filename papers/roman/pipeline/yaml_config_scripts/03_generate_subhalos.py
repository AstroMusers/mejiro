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


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
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
    if verbose: print(f'Reading from {input_dir}')

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if verbose: print(f'Set up output directory {output_dir}')

    # open pickled lens list
    pickles = glob(os.path.join(input_dir, f'{config["pipeline_dir"]}_detectable_lenses_sca*.pkl'))
    scas = [int(f.split('_')[-1].split('.')[0][3:]) for f in pickles]
    scas = sorted([str(sca).zfill(2) for sca in scas])
    for sca in scas:
        os.makedirs(os.path.join(output_dir, f'sca{sca}'), exist_ok=True)
    sca_dict = {}
    total = 0
    for sca in scas:
        pickle_path = os.path.join(input_dir, f'{config["pipeline_dir"]}_detectable_lenses_sca{sca}.pkl')
        lens_list = util.unpickle(pickle_path)
        sca_dict[sca] = lens_list
        total += len(lens_list)
    count = total
    if total == 0:
        raise FileNotFoundError(f'No pickled lenses found. Check {input_dir}.')
    print(f'Processing {total} lens(es)')

    # tuple the parameters
    tuple_list = []
    for sca, lens_list in sca_dict.items():
        sca_id = str(sca).zfill(2)
        sca_dir = os.path.join(output_dir, f'sca{sca_id}')
        for lens in lens_list:
            tuple_list.append((lens, subhalo_config, sca_dir))

    # implement limit if one exists
    if limit is not None:
        if limit > count:
            limit = count
        tuple_list = tuple_list[:limit]
        count = limit
        print(f'Limiting to {limit} lens(es)')

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
    (lens, subhalo_config, output_dir) = tuple

    # unpack pipeline_params
    los_normalization = subhalo_config['los_normalization']
    r_tidal = subhalo_config['r_tidal']
    sigma_sub = subhalo_config['sigma_sub']
    log_mlow = subhalo_config['log_mlow']
    log_mhigh = subhalo_config['log_mhigh']

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
        print(f'Failed to generate subhalos for lens {lens.name.split("_")[2]}: {e}')
        return

    # Add subhalos
    lens.add_realization(cdm_realization)

    # Pickle the subhalo realization
    subhalo_dir = os.path.join(output_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.name.split("_")[2]}.pkl'), cdm_realization)

    # Pickle the lens with subhalos
    pickle_target = os.path.join(output_dir, f'lens_with_subhalos_{lens.name.split("_")[2]}.pkl')
    util.pickle(pickle_target, lens)


if __name__ == '__main__':
    main()
