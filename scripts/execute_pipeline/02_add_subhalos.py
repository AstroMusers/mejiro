import multiprocessing
import os
import sys
import time
import traceback
from multiprocessing import Pool

import hydra
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()
    
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # directory to write the lenses with subhalos to
    output_dir = os.path.join(pickle_dir, '02_lenses_with_substructure')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # open pickled lens list
    lens_list = util.unpickle(os.path.join(pickle_dir, '01_skypy_output_lens_list'))

    # TEMP: optionally, grab the first handful
    # lens_list = lens_list[:100]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed

    # tuple the parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    tuple_list = []
    for i, _ in enumerate(lens_list):
        tuple_list.append((lens_list[i], pipeline_params, output_dir))

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    # lenses_with_substructure = []
    success_count = 0
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for updated_lens in pool.map(add, batch):
            if updated_lens is not None:
                success_count += 1
                # lenses_with_substructure.append(updated_lens)

    # for lens in tqdm(lens_list):
    #     updated_lens = add(lens)
    #     if updated_lens is not None:
    #         updated_lenses.append(updated_lens)

    # pickle lens list
    # pickle_target = os.path.join(pickle_dir, '02_skypy_output_lens_list_with_subhalos')
    # util.delete_if_exists(pickle_target)
    # util.pickle(pickle_target, lenses_with_substructure)
                
    print(f'Added subhalos to {success_count} of {len(lens_list)} lenses')

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

    # add CDM subhalos
    try:
        # use same CDM subhalos
        # lens.add_subhalos(*pyhalo.unpickle_subhalos('/data/bwedig/roman-pandeia/output/pickles/pyhalo/cdm_subhalos_tuple'))

        # TODO update with subhalo_cone param from hydra

        # randomly generate CDM subhalos
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source, cone_opening_angle_arcsec=subhalo_cone, LOS_normalization=los_normalization))
        
        # pickle
        pickle_target = os.path.join(output_dir, f'lens_with_subhalos_{lens.uid}')
        util.pickle(pickle_target, lens)
        return 1
    except:
        # print(traceback.format_exc())
        return None


if __name__ == '__main__':
    main()
