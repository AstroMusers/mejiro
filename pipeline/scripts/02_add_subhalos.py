import multiprocessing
import os
import sys
import time
from glob import glob
from multiprocessing import Pool

import hydra
import numpy as np
from astropy.cosmology import default_cosmology
from pyHalo.preset_models import CDM
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # open pickled lens list
    pickles = glob(os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lenses_sca*.pkl'))
    scas = [int(f.split('_')[-1].split('.')[0][3:]) for f in pickles]
    scas = sorted([str(sca).zfill(2) for sca in scas])
    sca_dict = {}
    total = 0
    for sca in scas:
        pickle_path = os.path.join(config.machine.dir_01, f'01_hlwas_sim_detectable_lenses_sca{sca}.pkl')
        lens_list = util.unpickle(pickle_path)
        sca_dict[sca] = lens_list
        total += len(lens_list)
    count = total
    assert total != 0, f'No pickled lenses found. Check {config.machine.dir_01}.'

    # directory to write the lenses with subhalos to
    output_parent_dir = config.machine.dir_02
    util.create_directory_if_not_exists(output_parent_dir)
    util.clear_directory(output_parent_dir)
    for sca in scas:
        os.makedirs(os.path.join(output_parent_dir, f'sca{sca}'), exist_ok=True)

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for sca, lens_list in sca_dict.items():
        sca_id = str(sca).zfill(2)
        output_dir = os.path.join(output_parent_dir, f'sca{sca_id}')
        for lens in lens_list:
            tuple_list.append((lens, pipeline_params, output_dir))

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(add, batch)

    stop = time.time()
    util.print_execution_time(start, stop)


def add(tuple):
    np.random.seed()

    from mejiro.utils import util

    # unpack tuple
    (lens, pipeline_params, output_dir) = tuple

    # unpack pipeline_params
    subhalo_cone = pipeline_params['subhalo_cone']
    los_normalization = pipeline_params['los_normalization']
    r_tidal = pipeline_params['r_tidal']
    sigma_sub = pipeline_params['sigma_sub']
    log_mlow = pipeline_params['log_mlow']
    log_mhigh = pipeline_params['log_mhigh']

    log_m_host = np.log10(lens.main_halo_mass)
    # kwargs_cosmo = util.get_kwargs_cosmo(lens.cosmo)  # TODO UPDATE 
    kwargs_cosmo = util.get_kwargs_cosmo(default_cosmology.get())
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
        print(f'Failed to generate subhalos for lens {lens.uid}: {e}')        
        return

    # add subhalos
    # stats_dict = lens.add_subhalos(cdm_realization, return_stats=True)
    lens.add_subhalos(cdm_realization)

    # pickle the stats
    # stats_dir = os.path.join(output_dir, 'stats')
    # util.create_directory_if_not_exists(stats_dir)
    # util.pickle(os.path.join(stats_dir, f'subhalo_stats_{lens.uid}.pkl'), stats_dict)

    # pickle the subhalo realization
    subhalo_dir = os.path.join(output_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.pickle(os.path.join(subhalo_dir, f'subhalo_realization_{lens.uid}.pkl'), cdm_realization)

    # pickle the lens with subhalos
    pickle_target = os.path.join(output_dir, f'lens_with_subhalos_{lens.uid}.pkl')
    util.pickle(pickle_target, lens)


if __name__ == '__main__':
    main()
