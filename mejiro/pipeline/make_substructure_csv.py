"""
Write a CSV of each system's substructure flag from existing step 03 output.

Usage:
    python3 make_substructure_csv.py --config <config.yaml> [--data_dir <output_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Optional override for the data directory specified in the config file.
"""
import argparse
import logging
import os
import time
from glob import glob

import pandas as pd
from tqdm import tqdm

from mejiro.analysis import lensing
from mejiro.utils import util
from mejiro.pipeline import config_schema
from mejiro.pipeline.pipeline_helper import PipelineHelper, load_config

logger = logging.getLogger(__name__)


def main(args):
    start = time.time()

    config = load_config(args.config)

    # standalone script (does not go through PipelineHelper), so configure logging and
    # validate the config here
    PipelineHelper.configure_logging_and_warnings(config)
    config_schema.validate(config, 'make_substructure_csv', source=args.config)

    data_dir = PipelineHelper.resolve_data_dir(config, args.data_dir)

    if config['dev']:
        config['pipeline_label'] += '_dev'

    input_dir = os.path.join(data_dir, config['pipeline_label'], '03')
    output_dir = os.path.join(data_dir, config['pipeline_label'])

    lens_pickles = sorted(glob(os.path.join(input_dir, 'sca*', 'lens_*.pkl')))
    logger.info(f'Found {len(lens_pickles)} system(s) in {input_dir}')

    # calculate and implement limit, if specified
    if config['limit'] is not None:
        lens_pickles = lens_pickles[:config['limit']]
        logger.info(f'Limiting to {len(lens_pickles)} systems')

    rows = []
    for path in tqdm(lens_pickles):
        lens = util.unpickle(path)
        realization = lens.realization

        system_id = lens.name

        if realization is None:
            # no substructure
            substructure_flag = None
        elif hasattr(lens, 'substructure_flag'):
            # lightweight serialization: realization stripped, flag stored at 03 time
            substructure_flag = lens.substructure_flag
        else:
            # full serialization: derive the flag from the realization object
            substructure_flag = lensing.substructure_flag(realization)

        rows.append({"id": system_id, "substructure_flag": substructure_flag})

    df = pd.DataFrame(rows, columns=["id", "substructure_flag"])
    logger.info(f'\n{df.head()}')

    csv_path = os.path.join(output_dir, "substructure_flags.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {csv_path}")

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Write a CSV of each system's substructure flag from step 03 output")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, default=None, help='Optional override for the data directory in the config.')
    args = parser.parse_args()
    main(args)
