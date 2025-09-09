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
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'roman_test.yaml')
        self.config_file = config_file

        parser = argparse.ArgumentParser(description="Run the mejiro pipeline")
        parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
        arg_list = ["--config", self.config_file]
        self.args = parser.parse_args(arg_list)

    def run(self):
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
