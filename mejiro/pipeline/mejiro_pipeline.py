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

    # TODO pipeline metrics
