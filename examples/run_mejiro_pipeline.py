import os

import mejiro
from mejiro.pipeline.mejiro_pipeline import Pipeline


def main():
    """
    Note that the `mejiro` pipeline takes a long time to run, especially when not parallelized. You may find a terminal multiplexer like `tmux` or `screen` useful.
    """
    # specify config file
    config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'roman_test.yaml')

    # run pipeline
    pipeline = Pipeline(config_file=config_file)
    pipeline.run()

    # the pipeline can be run step-by-step by calling the `run_script()` method; see the docstring for details

    # print metrics
    pipeline.print_metrics()


if __name__ == '__main__':
    main()
