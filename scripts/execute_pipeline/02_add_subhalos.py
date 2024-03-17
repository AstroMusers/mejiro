import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import hydra
import numpy as np
from tqdm import tqdm

from pyHalo.preset_models import CDM


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # open pickled lens list
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_skypy_output_lens_list')
    lens_list = util.unpickle(pickled_lens_list)
    assert len(lens_list) != 0, f'No pickled lenses found. Check {pickled_lens_list}.'
    count = len(lens_list)

    # directory to write the lenses with subhalos to
    output_dir = config.machine.dir_02
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # directory to write pickled subhalos to
    subhalo_dir = os.path.join(output_dir, 'subhalos')
    util.create_directory_if_not_exists(subhalo_dir)
    util.clear_directory(subhalo_dir)

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for lens in lens_list:
        tuple_list.append((lens, pipeline_params, output_dir))

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4
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
    from mejiro.utils import util

    # unpack tuple
    (lens, pipeline_params, output_dir) = tuple

    # unpack pipeline_params
    subhalo_cone = pipeline_params['subhalo_cone']
    los_normalization = pipeline_params['los_normalization']

    # circumvent bug with pyhalo, sometimes fails when redshifts have more than 2 decimal places
    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)    

    # randomly generate CDM subhalos
    log_m_host = np.log10(lens.lens_mass)
    # TODO calculate r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed in 3D with a cored NFW profile with this core radius; by default, it's 0.25
    r_tidal = 0.25
    sigma_sub = 0.055
    cdm_realization = CDM(z_lens, 
                          z_source, 
                          sigma_sub=sigma_sub,
                          log_mlow=6.,
                          log_mhigh=10.,
                          log_m_host=log_m_host, 
                          r_tidal=r_tidal,
                          cone_opening_angle_arcsec=subhalo_cone,
                          LOS_normalization=los_normalization)

    # add subhalos
    lens.add_subhalos(cdm_realization)

    # pickle the subhalo realization
    util.pickle(os.path.join(output_dir, 'subhalos', f'subhalo_realization_{lens.uid}'), cdm_realization)

    # pickle the lens with subhalos
    pickle_target = os.path.join(output_dir, f'lens_with_subhalos_{lens.uid}')
    util.pickle(pickle_target, lens)


if __name__ == '__main__':
    main()
