import argparse

from mejiro.pipeline.mejiro_pipeline import Pipeline


def main():
    """
    Run the pipeline with `simple.yaml`, which takes a few minutes. Configs that simulate more systems take much longer, especially when not parallelized, so you may find a terminal multiplexer like `tmux` or `screen` useful.
    """
    parser = argparse.ArgumentParser(description='Run the mejiro pipeline with simple.yaml')
    parser.add_argument('data_dir', help='Directory to write the pipeline output to')
    args = parser.parse_args()

    # run pipeline; to use your own config, pass it as Pipeline(config_file=...)
    pipeline = Pipeline(data_dir=args.data_dir)
    pipeline.run()

    # the pipeline can be run step-by-step by calling the `run_script()` method; see the docstring for details

    # print metrics
    pipeline.print_metrics()


if __name__ == '__main__':
    main()
