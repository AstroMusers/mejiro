import csv
import os
import sys
import time
from datetime import datetime
from glob import glob

import h5py
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
    from mejiro.lenses.strong_lens import StrongLens

    pipeline_params = util.hydra_to_dict(config.pipeline)
    pieces = pipeline_params['pieces']

    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    output_file = os.path.join(config.machine.pipeline_dir, f'mejiro_output_{now_string}.hdf5')
    output_csv = os.path.join(config.machine.pipeline_dir, f'mejiro_metadata_{now_string}.csv')

    color_image_paths = sorted(glob(f'{config.machine.dir_05}/*.npy'))
    image_paths = sorted(glob(f'{config.machine.dir_04}/*[0-9].npy'))
    subhalo_paths = sorted(glob(f'{config.machine.dir_02}/subhalos/*'))
    if pieces:
        isolated_source_paths = sorted(glob(f'{config.machine.dir_04}/*_source.npy'))
        isolated_lens_paths = sorted(glob(f'{config.machine.dir_04}/*_lens.npy'))

    # make sure correct number of files are identified
    assert len(color_image_paths) == len(image_paths) == len(subhalo_paths)
    if pieces:
        assert len(isolated_source_paths) == len(isolated_lens_paths) == len(color_image_paths)

    # load data
    color_images = [np.load(f) for f in color_image_paths]
    images = [np.load(f) for f in image_paths]
    subhalos = [util.unpickle(f) for f in subhalo_paths]
    if pieces:
        isolated_sources = [np.load(f) for f in isolated_source_paths]
        isolated_lenses = [np.load(f) for f in isolated_lens_paths]

    with h5py.File(output_file, 'w') as hf:
        # create datasets
        color_image_dataset = hf.create_dataset('color_images', data=color_images)
        image_dataset = hf.create_dataset('images', data=images)
        subhalo_dataset = hf.create_dataset('subhalos', data=subhalos)
        if pieces:
            isolated_source_dataset = hf.create_dataset('isolated_sources', data=isolated_sources)
            isolated_lens_dataset = hf.create_dataset('isolated_lenses', data=isolated_lenses)

        # set attributes
        hf.attrs['n_images'] = len(images)
        for key, value in util.hydra_to_dict(config.pipeline).items():
            hf.attrs[key] = value

        # TODO set attributes on each dataset

    print(f'Output .hdf5 file: {output_file}')

    # generate csv file
    lens_paths = glob(f'{config.machine.dir_03}/lens_*')
    lens_paths.sort()

    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(StrongLens.get_csv_headers())

        for lens_path in tqdm(lens_paths):
            lens = util.unpickle(lens_path)
            writer.writerow(lens.csv_row())

    print(f'Output .csv file: {output_csv}')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
