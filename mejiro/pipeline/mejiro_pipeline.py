import argparse
import os
import yaml

import mejiro
from mejiro.pipeline import (
    _00_cache_psfs as script_00_cache_psfs,
    _01_run_survey_simulation as script_01_run_survey_simulation,
    _02_build_lens_list as script_02_build_lens_list,
    _03_generate_subhalos as script_03_generate_subhalos,
    _04_create_synthetic_images as script_04_create_synthetic_images,
    _05_create_exposures as script_05_create_exposures,
    _06_h5_export as script_06_h5_export,
)

class Pipeline:
    """
    Class for running and managing the Mejiro pipeline. It wraps the individual pipeline scripts to provide a simple interface.

    Attributes
    ----------
    config_file : str, optional
        Path to the pipeline configuration YAML file. If None, a default config file is used.
    data_dir : str, optional
        Directory containing input data. If None, uses the value from the config file.
    _test_mode : bool, optional
        If True, skips execution of the first pipeline script (used for testing).
    """
    def __init__(self, config_file=None, data_dir=None, _test_mode=False):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'simple.yaml')
        self.config_file = config_file

        parser = argparse.ArgumentParser(description="Run the mejiro pipeline")
        parser.add_argument('--config')
        arg_list = ["--config", self.config_file]
        if data_dir is not None:
            parser.add_argument('--data_dir')
            arg_list += ["--data_dir", data_dir]
        self.args = parser.parse_args(arg_list)

        self._test_mode = _test_mode

    def run(self):
        """
        Executes the mejiro pipeline end-to-end.

        The pipeline consists of the following steps:
            1. Optionally caches PSFs if not in test mode.
            2. Runs the survey simulation.
            3. Builds the lens list.
            4. Generates subhalos.
            5. Creates synthetic images.
            6. Creates exposures.
            7. Exports results to HDF5 format.

        Returns
        -------
        None

        Notes
        -----
        The method assumes that each script's `main` function accepts `self.args` as its argument.
        The PSF caching step is skipped if `_test_mode` is True.
        """
        if not self._test_mode:
            script_00_cache_psfs.main(self.args)
        script_01_run_survey_simulation.main(self.args)
        script_02_build_lens_list.main(self.args)
        script_03_generate_subhalos.main(self.args)
        script_04_create_synthetic_images.main(self.args)
        script_05_create_exposures.main(self.args)
        script_06_h5_export.main(self.args)

    def run_script(self, script_number):
        """
        Run a specific pipeline script by its number.

        Parameters
        ----------
        script_number : int
            The script number to run (0-6).

        Script numbers and their corresponding scripts
        ---------------------------------------------
        0 : Cache psfs
        1 : Run survey simulation
        2 : Build lens list
        3 : Generate subhalos
        4 : Create synthetic images
        5 : Create exposures
        6 : Export to HDF5 file

        Raises
        ------
        ValueError
            If the script_number is not between 0 and 6.
        """
        if script_number == 0:
            script_00_cache_psfs.main(self.args)
        elif script_number == 1:
            script_01_run_survey_simulation.main(self.args)
        elif script_number == 2:
            script_02_build_lens_list.main(self.args)
        elif script_number == 3:
            script_03_generate_subhalos.main(self.args)
        elif script_number == 4:
            script_04_create_synthetic_images.main(self.args)
        elif script_number == 5:
            script_05_create_exposures.main(self.args)
        elif script_number == 6:
            script_06_h5_export.main(self.args)
        else:
            raise ValueError(f"Script number {script_number} is not valid. Please choose a number between 0 and 6.")

    # TODO run a subset of the pipeline

    def print_metrics(self):
        """
        Prints pipeline execution metrics including total execution time and average execution time per system.
        Loads configuration and execution times from files, calculates the total pipeline execution time,
        and prints it in human-readable format (hours, minutes, seconds, and days if applicable). Also computes
        and prints the average execution time per system based on the number of pickled exposure files found.
        Raises
        ------
        ValueError
            If no pickled exposures are found in the expected directory.
        Notes
        -----
        - Assumes the existence of a configuration file and an execution_times.json file.
        - The method expects the configuration to specify the data directory and band information.
        - The method modifies the pipeline label if running in development mode.
        """
        import datetime
        import json
        from glob import glob
        from pprint import pprint

        with open(self.config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        if config['dev']:
            config['pipeline_label'] += '_dev'

        with open(os.path.join(config['data_dir'], config['pipeline_label'], 'execution_times.json'), 'r') as f:
            execution_times = json.load(f)

        pprint(execution_times)

        total_time = 0
        for script_name, times in execution_times.items():
            h, m, s = times.split(':')
            if 'days' in h:
                d, h = h.split('days, ')
                h = int(d) * 24 + int(h)
            elif 'day' in h:
                d, h = h.split('day, ')
                h = int(d) * 24 + int(h)
            time = (int(h) * 3600) + (int(m) * 60) + int(s)
            total_time += time

        print(
            f'Total pipeline execution time: {datetime.timedelta(seconds=total_time)}, {total_time} seconds, {total_time / 3600:.2f} hours')
        
        data_dir = os.path.join(config['data_dir'], config['pipeline_label'], '05')

        band = config['synthetic_image']['bands'][0]
        pickles = sorted(glob(os.path.join(data_dir, 'sca*', f'Exposure_*_{band}.pkl')))
        # pickles = sorted(glob(os.path.join(data_dir, 'Exposure_*.pkl')))

        if len(pickles) > 0:
            print(f'Execution time per system: {total_time / len(pickles):.3f} seconds')
        else:
            raise ValueError(f'No pickled exposures found in {data_dir}')
