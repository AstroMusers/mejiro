import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool

import hydra
from tqdm import tqdm

from package.helpers import pyhalo
from package.utils import util


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

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
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        for i, output in enumerate(pool.map(add_subhalos, batch)):
            (lens) = output
            if lens is not None:
                lens_list.append(lens)

    # pickle lens list
    pickle_target = os.path.join(pickle_dir, '02_skypy_output_lens_list_with_subhalos')
    util.delete_if_exists(pickle_target)
    with open(pickle_target, 'ab') as results_file:
        pickle.dump(lens_list, results_file)


def add_subhalos(lens):
    # add CDM subhalos
    try:
        lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))
    except:
        return None


if __name__ == '__main__':
    main()
