import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool
import time
import datetime

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

    # open pickled lens list
    with open(os.path.join(pickle_dir, '01_skypy_output_lens_list'), 'rb') as results_file:
        lens_list = pickle.load(results_file)

    # TODO TEMP: for now, just grab the first handful
    # lens_list = lens_list[:100]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    generator = util.batch_list(lens_list, process_count)
    batches = list(generator)

    # process the batches
    updated_lenses = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for updated_lens in pool.map(add, batch):
            if updated_lens is not None:
                updated_lenses.append(updated_lens)

    # for lens in tqdm(lens_list):
    #     updated_lens = add(lens)
    #     if updated_lens is not None:
    #         updated_lenses.append(updated_lens)

    # pickle lens list
    pickle_target = os.path.join(pickle_dir, '02_skypy_output_lens_list_with_subhalos')
    util.delete_if_exists(pickle_target)
    with open(pickle_target, 'ab') as results_file:
        pickle.dump(updated_lenses, results_file)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def add(lens):
    from mejiro.helpers import pyhalo
    # add CDM subhalos
    try:
        # use same CDM subhalos
        # lens.add_subhalos(*pyhalo.unpickle_subhalos('/data/bwedig/roman-pandeia/output/pickles/pyhalo/cdm_subhalos_tuple'))

        # randomly generate CDM subhalos
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source, cone_opening_angle_arcsec=10, LOS_normalization=0.0))
        return lens
    except:
        return None


if __name__ == '__main__':
    main()
