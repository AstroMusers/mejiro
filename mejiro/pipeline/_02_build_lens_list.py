"""
Builds a list of StrongLens objects from previously detected lensing systems.

This script processes the output of a strong lensing survey simulation, converting detected lensing systems into mejiro StrongLens objects. It reads configuration parameters from a mejiro configuration YAML file and supports multiple instruments (Roman, HWO). Output lenses are pickled for downstream analysis.

Usage:
    python3 _02_build_lens_list.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import argparse
import os
import time
from glob import glob
from tqdm import tqdm

from mejiro.galaxy_galaxy import GalaxyGalaxy
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper


PREV_SCRIPT_NAME = '01'
SCRIPT_NAME = '02'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst']


def main(args):
    start = time.time()

    # initialize PipeLineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # retrieve configuration parameters
    use_jax = pipeline.config['jaxtronomy']['use_jax']
    scas = pipeline.config['survey']['detectors']
    bands = pipeline.config['survey']['bands']

    # tell script where the output of previous script is
    detectable_gglens_pickles = sorted(glob(pipeline.input_dir + '/detectable_gglenses_*.pkl'))
    if len(detectable_gglens_pickles) == 0:
        raise FileNotFoundError(f'No output files found. Check simulation output directory ({pipeline.input_dir}).')
    if pipeline.instrument_name == 'roman':
        parsed_names = [os.path.basename(f).split("_")[3].split(".")[0] for f in detectable_gglens_pickles]
        scas = [int(d[3:]) for d in parsed_names]
        scas = sorted(set([str(sca).zfill(2) for sca in scas]))
        if pipeline.verbose: print(f'Found SCA(s): {scas}')
    elif pipeline.instrument_name == 'hwo' or pipeline.instrument_name == 'jwst':
        pass
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # set up output directories
    if pipeline.instrument_name == 'roman':
        _ = pipeline.create_roman_sca_output_directories()

    uid = 0
    if pipeline.instrument_name == 'roman':
        for sca in tqdm(scas, desc="SCAs", position=0, leave=True):
            sca_pickles = [f for f in detectable_gglens_pickles if f'sca{sca}' in f]

            for pickled_list in tqdm(sca_pickles, desc="Runs", position=1, leave=False):
                gglenses = util.unpickle(pickled_list)

                for slsim_lens in tqdm(gglenses, desc="Strong Lenses", position=2, leave=False):
                    mejiro_lens = GalaxyGalaxy.from_slsim(slsim_lens, name=f'{pipeline.name}_{str(uid).zfill(8)}', bands=bands, use_jax=use_jax)
                    mejiro_lens_pickle_target = os.path.join(pipeline.output_dir, f'sca{sca}/lens_{mejiro_lens.name}.pkl')
                    util.pickle(mejiro_lens_pickle_target, mejiro_lens)
                    uid += 1

                if uid == pipeline.limit:
                    break

    elif pipeline.instrument_name == 'hwo' or pipeline.instrument_name == 'jwst':
        for pickled_list in tqdm(detectable_gglens_pickles, desc="Runs", position=1, leave=False):
            gglenses = util.unpickle(pickled_list)

            for slsim_lens in tqdm(gglenses, desc="Strong Lenses", position=2, leave=False):
                mejiro_lens = GalaxyGalaxy.from_slsim(slsim_lens, name=f'{pipeline.name}_{str(uid).zfill(8)}', use_jax=use_jax)
                mejiro_lens_pickle_target = os.path.join(pipeline.output_dir, f'lens_{mejiro_lens.name}.pkl')
                util.pickle(mejiro_lens_pickle_target, mejiro_lens)
                uid += 1

            if uid == pipeline.limit:
                break

    if pipeline.verbose: print(f'Pickled {uid} lens(es) to {pipeline.output_dir}')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME, os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build mejiro StrongLens objects.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False, help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    args = parser.parse_args()
    main(args)
