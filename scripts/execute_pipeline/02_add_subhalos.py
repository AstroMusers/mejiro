import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import hydra
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()
    
    repo_dir, pickle_dir = config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - 4

    # get bands
    bands = util.hydra_to_dict(config.pipeline)['band']

    # organize the pickles into a dict
    lens_dict = {}
    for band in bands:
        # open pickled lens list
        lens_list = util.unpickle(os.path.join(dir_01, f'01_skypy_output_lens_list_{band.lower()}'))
        lens_dict[band] = lens_list
    
    # TODO this naming is dumb, fix it
    lenses = []
    # create a tuple for each lens
    for i, _ in enumerate(lens_list):
        lens = {}
        for band in bands:
            lens[band] = lens_dict[band][i]
        lenses.append(lens)

    # directory to write the lenses with subhalos to
    output_dir = os.path.join(pickle_dir, '02_lenses_with_substructure')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for i, _ in enumerate(lenses):
        tuple_list.append((lenses[i], pipeline_params, output_dir))

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
    from mejiro.helpers import pyhalo
    from mejiro.utils import util

    # unpack tuple
    (lens, pipeline_params, output_dir) = tuple

    # unpack pipeline_params
    subhalo_cone = pipeline_params['subhalo_cone']
    los_normalization = pipeline_params['los_normalization']
   
    z_lens = round(list(lens.values())[0].z_lens, 2)
    z_source = round(list(lens.values())[0].z_source, 2)

    # randomly generate CDM subhalos
    halo_tuple = pyhalo.generate_CDM_halos(z_lens, z_source, cone_opening_angle_arcsec=subhalo_cone, LOS_normalization=los_normalization)

    # print(halo_tuple)

    # add this subhalo population to the lens for each filter
    for band, band_lens in lens.items():
        band_lens.add_subhalos(*halo_tuple)

        pickle_target = os.path.join(output_dir, f'lens_with_subhalos_{band_lens.uid}_{band.lower()}')
        util.pickle(pickle_target, band_lens)


if __name__ == '__main__':
    main()
