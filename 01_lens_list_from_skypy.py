import os
import pickle
import sys

import hydra
import pandas as pd
from tqdm import tqdm

from package.lenses.lens import Lens
from package.utils import util


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir, pickle_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir, config.machine.pickle_dir
    util.create_directory_if_not_exists(pickle_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get output of SkyPy pipeline
    df = pd.read_csv(os.path.join('/data', 'bwedig', 'roman-population', 'data', 'dictparaggln_Area00000010.csv'))

    lens_list = []

    # TODO implement multiprocessing to parallelize
    # generate the lens objects
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        lens_list.append(generate_lens(row))

    # # split up the rows into batches based on core count
    # cpu_count = multiprocessing.cpu_count()
    # process_count = int(cpu_count / 2)  # TODO consider making this larger, but using all CPUs has crashed
    # generator = util.batch_list(list(df.iterrows()), process_count)
    # batches = list(generator)

    # # process the batches
    # for batch in tqdm(batches):
    #     pool = Pool(processes=process_count) 
    #     for i, output in enumerate(pool.map(generate_lens, batch)):
    #         (lens) = output
    #         lens_list.append(lens)

    # pickle lens list
    with open(os.path.join(pickle_dir, 'skypy_output_lens_list'), 'ab') as results_file:
        pickle.dump(lens_list, results_file)


def generate_lens(row):
    return Lens(z_lens=row['redslens'],
                z_source=row['redssour'],
                theta_e=row['angleins'],
                lens_x=row['xposlens'],
                lens_y=row['yposlens'],
                source_x=row['xpossour'],
                source_y=row['ypossour'],
                mag_lens=row['magtlensF106'],
                mag_source=row['magtsourF106'])


if __name__ == '__main__':
    main()
