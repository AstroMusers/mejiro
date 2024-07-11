import os
import sys
import time
from glob import glob

import hydra
import numpy as np
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

    pipeline_params = util.hydra_to_dict(config.pipeline)
    limit = pipeline_params['limit']

    output_files = glob(config.machine.dir_00 + '/detectable_pop_*.csv')
    assert len(
        output_files) != 0, f'No output files found. Check HLWAS simulation output directory ({config.machine.dir_00}).'
    num_runs = len(output_files)

    uid = 0
    lens_list = []
    for run in range(num_runs):
        print(f'Run {run + 1} of {num_runs}')
        # unpickle the lenses from the population survey and create lens objects
        lens_paths = glob(config.machine.dir_00 + f'/run_{str(run).zfill(2)}/detectable_lens_{str(run).zfill(2)}_*.pkl')
        # TODO this fails for small survey areas where no lenses are expected
        # assert len(
        #     lens_paths) != 0, f'No pickled lenses found. Check SkyPy output directory {config.machine.dir_00}.'

        for _, lens in tqdm(enumerate(lens_paths), total=len(lens_paths)):
            lens = lens_util.unpickle_lens(lens, str(uid).zfill(8))
            uid += 1
            lens_list.append(lens)

    # TODO this isn't the most efficient way of doing this, but these operations aren't terribly slow so I can get away with it, but also inefficient code makes me sad
    # lens_list = lens_list[:limit]
    lens_list = np.random.choice(lens_list, size=limit)

    # pickle lens list
    pickle_target = os.path.join(config.machine.dir_01, f'01_hlwas_sim_detectable_lens_list.pkl')
    util.delete_if_exists(pickle_target)
    util.pickle(pickle_target, lens_list)

    print(f'Pickled {len(lens_list)} lenses to {pickle_target}')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
