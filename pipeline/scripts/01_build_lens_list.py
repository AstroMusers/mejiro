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

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    survey_params = util.hydra_to_dict(config.survey)
    debugging = pipeline_params['debugging']
    limit = pipeline_params['limit']
    scas = survey_params['scas']

    if debugging:
        input_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '00')
        output_dir = os.path.join(f'{config.machine.pipeline_dir}_dev', '01')
    else:
        input_dir = config.machine.dir_00
        output_dir = config.machine.dir_01
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if debugging: 
        print(f'Reading from {input_dir}')
        print(f'Set up output directory {output_dir}')

    # make sure there are files to process
    input_files = glob(input_dir + '/detectable_pop_*.csv')
    assert len(
        input_files) != 0, f'No output files found. Check HLWAS simulation output directory ({input_dir}).'

    uid = 0
    for sca_id in tqdm(scas, disable=not debugging):
        sca_id = str(sca_id).zfill(2)

        csvs = [f for f in input_files if f'sca{sca_id}' in f]
        
        if len(csvs) == 0:
            continue

        # get all runs associated with this SCA
        runs = [int(f.split('_')[-2]) for f in csvs]  # TODO I don't love string parsing that relies on file naming conventions
    
        lens_list = []
        for _, run in enumerate(runs):
            # unpickle the lenses from the population survey and create lens objects
            lens_paths = glob(input_dir + f'/run_{str(run).zfill(4)}_sca{sca_id}/detectable_lens_{str(run).zfill(4)}_*.pkl')
            
            if len(lens_paths) == 0:
                continue

            for _, lens_path in enumerate(lens_paths):
                lens = lens_util.unpickle_lens(lens_path, str(uid).zfill(8))
                uid += 1
                lens_list.append(lens)
            
            if uid == limit:
                break

        pickle_target = os.path.join(output_dir, f'01_hlwas_sim_detectable_lenses_sca{sca_id}.pkl')
        util.pickle(pickle_target, lens_list)
    
    print(f'Pickled {uid} lenses to {output_dir}')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
