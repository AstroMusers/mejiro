import argparse
import os

import mejiro
from mejiro.pipeline.mejiro_pipeline import Pipeline


def main(args):
    """
    Note that the `mejiro` pipeline takes a long time to run, especially when not parallelized. You may find a terminal multiplexer like `tmux` or `screen` useful.
    """

    # ensure the configuration file has a .yaml or .yml extension
    if not args.config.endswith(('.yaml', '.yml')):
        if os.path.exists(args.config + '.yaml'):
            args.config += '.yaml'
        elif os.path.exists(args.config + '.yml'):
            args.config += '.yml'
        else:
            raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")

    # set configuration file
    if args.config is not None:
        config_file = args.config
    else:
        config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'roman_test.yaml')

    # run pipeline
    pipeline = Pipeline(config_file)
    pipeline.run()

    # the pipeline can be run step-by-step by calling the `run_script()` method; see the docstring for details

    # print metrics
    pipeline.print_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run mejiro pipeline")
    parser.add_argument('--config', type=str, required=False, help='Filepath of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
