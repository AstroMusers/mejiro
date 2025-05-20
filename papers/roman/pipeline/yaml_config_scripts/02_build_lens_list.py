import os
import sys
import time
import yaml
from glob import glob
from tqdm import tqdm


PREV_SCRIPT_NAME = '01'
SCRIPT_NAME = '02'


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.galaxy_galaxy import GalaxyGalaxy
    from mejiro.utils import util

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    limit = config['limit']
    scas = config['survey']['scas']

    # set up top directory for all pipeline output
    pipeline_dir = os.path.join(data_dir, config['pipeline_dir'])
    if dev:
        pipeline_dir += '_dev'

    # tell script where the output of previous script is
    input_dir = os.path.join(pipeline_dir, PREV_SCRIPT_NAME)
    if verbose: print(f'Reading from {input_dir}')

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if verbose: print(f'Set up output directory {output_dir}')

    # make sure there are files to process
    input_files = glob(input_dir + '/detectable_pop_*.csv')
    if len(input_files) == 0:
        raise FileNotFoundError(f'No output files found. Check HLWAS simulation output directory ({input_dir}).')

    uid = 0
    for sca in tqdm(scas, desc="SCAs", position=0, leave=True):
        sca = str(sca).zfill(2)

        csvs = [f for f in input_files if f'sca{sca}' in f]

        if len(csvs) == 0:
            continue

        # get all runs associated with this SCA
        runs = [int(f.split('_')[-2]) for f in
                csvs]  # TODO I don't love string parsing that relies on file naming conventions

        lens_list = []
        for run in tqdm(runs, desc="Runs", position=1, leave=False):
            # unpickle the lenses from the population survey and create lens objects
            lens_paths = glob(
                input_dir + f'/run_{str(run).zfill(4)}_sca{sca}/detectable_lens_{str(run).zfill(4)}_*.pkl')

            if len(lens_paths) == 0:
                continue

            for lens_path in tqdm(lens_paths, desc="Strong Lenses", position=2, leave=False):
                slsim_lens = util.unpickle(lens_path)
                mejiro_lens = GalaxyGalaxy.from_slsim(slsim_lens, name=f'{config["pipeline_dir"]}_{str(uid).zfill(8)}')
                uid += 1
                lens_list.append(mejiro_lens)

            if uid == limit:
                break

        pickle_target = os.path.join(output_dir, f'{config["pipeline_dir"]}_detectable_lenses_sca{sca}.pkl')
        util.pickle(pickle_target, lens_list)

    if verbose: print(f'Pickled {uid} lenses to {output_dir}')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME, os.path.join(pipeline_dir, 'execution_times.json'))


if __name__ == '__main__':
    main()
