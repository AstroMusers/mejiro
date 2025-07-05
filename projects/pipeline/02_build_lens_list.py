import argparse
import os
import sys
import time
import yaml
from glob import glob
from tqdm import tqdm


PREV_SCRIPT_NAME = '01'
SCRIPT_NAME = '02'


def main(args):
    start = time.time()

    # ensure the configuration file has a .yaml or .yml extension
    if not args.config.endswith(('.yaml', '.yml')):
        if os.path.exists(args.config + '.yaml'):
            args.config += '.yaml'
        elif os.path.exists(args.config + '.yml'):
            args.config += '.yml'
        else:
            raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")

    # read configuration file
    with open(args.config, 'r') as f:
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
    detectable_gglens_pickles = sorted(glob(input_dir + '/detectable_gglenses_*.pkl'))
    if len(detectable_gglens_pickles) == 0:
        raise FileNotFoundError(f'No output files found. Check simulation output directory ({input_dir}).')
    parsed_names = [os.path.basename(f).split("_")[3].split(".")[0] for f in detectable_gglens_pickles]
    scas = [int(d[3:]) for d in parsed_names]
    scas = sorted(set([str(sca).zfill(2) for sca in scas]))
    if verbose: print(f'Found SCA(s): {scas}')

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    output_sca_dirs = []
    for sca in scas:
        sca_dir = os.path.join(output_dir, f'sca{sca}')
        os.makedirs(sca_dir, exist_ok=True)
        output_sca_dirs.append(sca_dir)
    if verbose: print(f'Set up output directories {output_sca_dirs}')  

    uid = 0
    for sca in tqdm(scas, desc="SCAs", position=0, leave=True):
        sca_pickles = [f for f in detectable_gglens_pickles if f'sca{sca}' in f]

        for pickled_list in tqdm(sca_pickles, desc="Runs", position=1, leave=False):
            gglenses = util.unpickle(pickled_list)

            for slsim_lens in tqdm(gglenses, desc="Strong Lenses", position=2, leave=False):
                mejiro_lens = GalaxyGalaxy.from_slsim(slsim_lens, name=f'{config["pipeline_dir"]}_{str(uid).zfill(8)}')
                mejiro_lens_pickle_target = os.path.join(output_dir, f'sca{sca}/lens_{str(uid).zfill(8)}.pkl')
                util.pickle(mejiro_lens_pickle_target, mejiro_lens)
                uid += 1

            if uid == limit:
                break

    if verbose: print(f'Pickled {uid} lens(es) to {output_dir}')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME, os.path.join(pipeline_dir, 'execution_times.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and cache Roman PSFs.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
