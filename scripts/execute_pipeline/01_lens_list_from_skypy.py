import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool

import hydra
import pandas as pd
from tqdm import tqdm
from glob import glob


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util

    util.create_directory_if_not_exists(pickle_dir)

    # get output of SkyPy pipeline
    # df = pd.read_csv(os.path.join('/data', 'bwedig', 'roman-population', 'data', 'dictparaggln_Area00000010.csv'))
        
    # unpickle the lenses from the population survey
    lens_dir = os.path.join('/data', 'bwedig', 'roman-population', 'data', 'lenses')
    lens_paths = glob(lens_dir + '/*')
    lens_list = [lens_util.unpickle_lens(i) for i in lens_paths]

    # split up the rows into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    generator = util.batch_list(lens_list, process_count)
    batches = list(generator)

    # process the batches
    updated_lenses = []
    for batch in tqdm(batches):
        pool = Pool(processes=process_count) 
        for updated_lens in pool.map(generate_lens, batch):
            updated_lenses.append(updated_lens)

    # pickle lens list
    pickle_target = os.path.join(pickle_dir, '01_skypy_output_lens_list')
    util.delete_if_exists(pickle_target)
    with open(pickle_target, 'ab') as results_file:
        pickle.dump(updated_lenses, results_file)


def generate_lens(lens):
    lens.get_array(num_pix=51, side=5.61)

    return lens


if __name__ == '__main__':
    main()
