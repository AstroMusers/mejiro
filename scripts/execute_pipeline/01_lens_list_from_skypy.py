import os
import sys
import time

import hydra
from tqdm import tqdm
from glob import glob


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir, pickle_dir = config.machine.repo_dir, config.machine.pickle_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util

    util.create_directory_if_not_exists(pickle_dir)
        
    for band in util.hydra_to_dict(config.pipeline)['band']:
        # unpickle the lenses from the population survey and create lens objects
        lens_dir = os.path.join('/data', 'bwedig', 'roman-population', 'data', 'lenses')
        lens_paths = glob(lens_dir + f'/*{band}*')
        lens_list = []
        for i, lens in tqdm(enumerate(lens_paths), total=len(lens_paths)):
            lens = lens_util.unpickle_lens(lens, str(i).zfill(8), band)
            lens_list.append(lens)

        # pickle lens list
        pickle_target = os.path.join(pickle_dir, f'01_skypy_output_lens_list_{band.lower()}')
        util.delete_if_exists(pickle_target)
        util.pickle(pickle_target, lens_list)

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
