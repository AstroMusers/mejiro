import os
import sys
import time
from glob import glob

import hydra
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util

    # create directory that this script will write to
    util.create_directory_if_not_exists(config.machine.dir_01)
    util.clear_directory(config.machine.dir_01)

    output_files = glob(config.machine.skypy_dir + '/skypy_output_*.csv')
    assert len(output_files) != 0, f'No output files found. Check SkyPy output directory ({config.machine.skypy_dir}).'
    num_runs = len(output_files)

    uid = 0
    lens_list = []
    for run in range(num_runs):
        print(f'Run {run + 1} of {num_runs}')
        # unpickle the lenses from the population survey and create lens objects
        lens_paths = glob(config.machine.skypy_dir + f'/lenses_5_run{str(run).zfill(3)}/*.pkl')
        assert len(
            lens_paths) != 0, f'No pickled lenses found. Check SkyPy output directory ({config.machine.skypy_dir}).'

        for _, lens in tqdm(enumerate(lens_paths), total=len(lens_paths)):
            lens = lens_util.unpickle_lens(lens, str(uid).zfill(8))
            uid += 1
            lens_list.append(lens)

    # pickle lens list
    pickle_target = os.path.join(config.machine.dir_01, f'01_skypy_output_lens_list')
    util.delete_if_exists(pickle_target)
    util.pickle(pickle_target, lens_list)

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
