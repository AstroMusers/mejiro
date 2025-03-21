import hydra
import multiprocessing
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pyHalo.preset_models import CDM
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir

    # Enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # Retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']

    # Set nice level
    os.nice(pipeline_params['nice'])

    # Set up input and output directories
    if debugging:
        input_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '01')
        output_parent_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '02')
    else:
        input_dir = config.machine.dir_01
        output_parent_dir = config.machine.dir_02
    util.create_directory_if_not_exists(output_parent_dir)
    util.clear_directory(output_parent_dir)

    # Open pickled lens list
    pickles = glob(os.path.join(input_dir, '01_hlwas_sim_detectable_lenses_sca*.pkl'))
    scas = [int(f.split('_')[-1].split('.')[0][3:]) for f in pickles]
    scas = sorted([str(sca).zfill(2) for sca in scas])
    for sca in scas:
        os.makedirs(os.path.join(output_parent_dir, f'sca{sca}'), exist_ok=True)
    sca_dict = {}
    total = 0
    for sca in scas:
        pickle_path = os.path.join(input_dir, f'01_hlwas_sim_detectable_lenses_sca{sca}.pkl')
        lens_list = util.unpickle(pickle_path)
        sca_dict[sca] = lens_list
        total += len(lens_list)
    count = total
    assert total != 0, f'No pickled lenses found. Check {input_dir}.'
    print(f'Processing {total} lens(es)')

    # Tuple the parameters
    tuple_list = []
    for sca, lens_list in sca_dict.items():
        sca_id = str(sca).zfill(2)
        output_dir = os.path.join(output_parent_dir, f'sca{sca_id}')
        for lens in lens_list:
            tuple_list.append((lens, pipeline_params, output_dir))

    # Implement limit if one exists
    if pipeline_params['limit'] != "None":  # TODO will need to be changed from string once yaml config implemented
        limit = pipeline_params['limit']
        if limit > count:
            limit = count
        tuple_list = tuple_list[:limit]
        count = limit
        print(f'Limiting to {limit} lens(es)')

    # Define the number of processes
    cpu_count = multiprocessing.cpu_count()
    process_count = config.machine.num_cores
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # Submit tasks to the executor
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(add, task): task for task in tuple_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Get the result to propagate exceptions if any

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '02',
                              os.path.join(os.path.dirname(output_parent_dir), 'execution_times.json'))


def add(tuple):
    np.random.seed()

    from mejiro.utils import util

    # Unpack tuple
    (lens, pipeline_params, output_dir) = tuple

    # Unpack pipeline_params
    los_normalization = pipeline_params['los_normalization']
    r_tidal = pipeline_params['r_tidal']
    sigma_sub = pipeline_params['sigma_sub']
    log_mlow = pipeline_params['log_mlow']
    log_mhigh = pipeline_params['log_mhigh']

    log_m_host = np.log10(lens.main_halo_mass)
    kwargs_cosmo = util.get_kwargs_cosmo(lens.cosmo)

    # Circumvent bug with pyhalo, sometimes fails when redshifts have more than 2 decimal places
    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)

    # Get Einstein radius
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
        print(f'Failed to generate subhalos for lens {lens.uid}: {e}')
        return

    # Add subhalos
    lens.add_subhalos(cdm_realization)

    # Pickle the subhalo realization
    subhalo_dir = os.path.join(output_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.uid}.pkl'), cdm_realization)

    # Pickle the lens with subhalos
    pickle_target = os.path.join(output_dir, f'lens_with_subhalos_{lens.uid}.pkl')
    util.pickle(pickle_target, lens)


if __name__ == '__main__':
    main()
