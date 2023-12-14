import multiprocessing
import os
import sys
from copy import deepcopy
from multiprocessing import Pool

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

from mejiro.helpers import pyhalo
from mejiro.lenses.lens import Lens
from mejiro.scripts import generate
from mejiro.utils import util


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def main(config):
    array_dir, data_dir, repo_dir = config.machine.array_dir, config.machine.data_dir, config.machine.repo_dir
    array_dir = os.path.join(array_dir, 'multiprocessing')
    util.create_directory_if_not_exists(array_dir)

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)

    # get path to Roman instrument params CSV
    roman_params_csv = os.path.join(repo_dir, 'data', 'roman_spacecraft_and_instrument_parameters.csv')

    # get output of SkyPy pipeline
    df = pd.read_csv(os.path.join('/data', 'bwedig', 'roman-population', 'data', 'dictparaggln_Area00000010.csv'))

    limit = 200
    total = deepcopy(limit)
    lens_list = []

    # generate the lens objects
    for i, row in tqdm(df.iterrows(), total=total):
        if i == limit:
            break

        lens = Lens(z_lens=row['redslens'],
                    z_source=row['redssour'],
                    sigma_v=row['velodisp'],
                    lens_x=row['xposlens'],
                    lens_y=row['yposlens'],
                    source_x=row['xpossour'],
                    source_y=row['ypossour'],
                    mag_lens=row['magtlensF106'],
                    mag_source=row['magtsourF106'])

        # add CDM subhalos
        try:
            lens.add_subhalos(*pyhalo.generate_CDM_halos(lens.z_lens, lens.z_source))
        except:
            continue

        lens_list.append(lens)

    # split the images into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)
    generator = util.batch_list(lens_list, process_count)
    batches = list(generator)

    image_list, execution_times, num_point_sources = [], [], []

    # process the batches
    for batch_index, batch in tqdm(enumerate(batches)):
        pool = Pool(processes=process_count)
        for i, output in enumerate(pool.map(generate.main, batch)):
            (image, execution_time, num_ps) = output
            np.save(os.path.join(array_dir, f'skypy_output_{batch_index}_{str(i).zfill(3)}'), image)
            execution_times.append(execution_time)
            num_point_sources.append(num_ps)

            # save other output lists
    np.save(os.path.join(array_dir, 'skypy_output_execution_times.npy'), execution_times)
    np.save(os.path.join(array_dir, 'skypy_output_num_point_sources.npy'), num_point_sources)


if __name__ == '__main__':
    main()
