import h5py
import os
import sys
import time
from datetime import datetime
from glob import glob

import hydra
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    output_file = os.path.join(config.machine.pipeline_dir, f'mejiro_output_{now_string}.hdf5')

    image_paths = glob(f'{config.machine.dir_05}/*.npy')
    subhalo_paths = glob(f'{config.machine.dir_02}/subhalos/*')

    # sort, so they're sorted in order by lens.uid
    image_paths.sort()
    subhalo_paths.sort()

    images = [np.load(f) for f in image_paths]
    subhalos = [util.unpickle(f) for f in subhalo_paths]

    with h5py.File(output_file, 'w') as hf:
        # create datasets
        image_dataset = hf.create_dataset('images', data=images)
        # subhalo_dataset = hf.create_dataset('subhalos', data=subhalos)

        # set attributes
        hf.attrs['n_images'] = len(images)
        for key, value in util.hydra_to_dict(config.pipeline).items():
            hf.attrs[key] = value

    print(f'Output file: {output_file}')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
