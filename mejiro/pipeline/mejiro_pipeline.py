import argparse
import copy
import logging
import os
import time
import yaml

import mejiro
from mejiro.utils import util

logger = logging.getLogger(__name__)
from mejiro.pipeline import (
    _00_cache_psfs as script_00_cache_psfs,
    _01a_generate_galaxy_tables as script_01a_generate_galaxy_tables,
    _01b_run_survey_simulation as script_01b_run_survey_simulation,
    _02_build_lens_list as script_02_build_lens_list,
    _03_generate_subhalos as script_03_generate_subhalos,
    _04_create_synthetic_images as script_04_create_synthetic_images,
    _05_galsim as script_05_galsim,
    _06_h5_export as script_06_h5_export,
    _05_romanisim as script_05_romanisim,
    calculate_snrs as script_calculate_snrs,
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
            config_name = 'test.yaml' if _test_mode else 'simple.yaml'
            config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', config_name)
        self.config_file = config_file

        parser = argparse.ArgumentParser(description="Run the mejiro pipeline")
        parser.add_argument('--config')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--sequential', action='store_true', default=False)
        parser.add_argument('--data_dir', default=None)
        # _05_romanisim flags (see its argparse for semantics); registered here so the
        # dispatched Namespace is complete for every script
        parser.add_argument('--level', choices=['l2', 'l3'], default='l2')
        parser.add_argument('--dither-pattern', dest='dither_pattern',
                            default=script_05_romanisim.DEFAULT_PATTERN)
        parser.add_argument('--max-systems', dest='max_systems', type=int, default=None)
        # _04_create_synthetic_images flag: which step to read lens pickles from
        parser.add_argument('--prev-step', dest='prev_step', choices=['02', '03'], default='03')
        arg_list = ["--config", self.config_file]
        if data_dir is not None:
            arg_list += ["--data_dir", data_dir]
        self.args = parser.parse_args(arg_list)

        self._test_mode = _test_mode

        # load config so dispatch flags are available
        with open(self.config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        # imaging.engine selects the exposure/export tail; step 04 reads
        # jaxtronomy.use_jax itself.
        self._engine = self.config['imaging']['engine']

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
        start = time.time()
        if not self._test_mode:
            script_00_cache_psfs.main(self.args)
        script_01a_generate_galaxy_tables.main(self.args)
        script_01b_run_survey_simulation.main(self.args)
        script_02_build_lens_list.main(self.args)
        script_03_generate_subhalos.main(self.args)
        script_04_create_synthetic_images.main(self.args)
        self._run_exposure_export_tail()
        stop = time.time()
        util.print_execution_time(start, stop)

    def _run_exposure_export_tail(self):
        """Run the exposure and HDF5-export steps for the configured ``imaging.engine``.

        * ``galsim``    -> _05_galsim -> _06_h5_export
        * ``romanisim`` -> _05_romanisim -> calculate_snrs -> _06_h5_export
        """
        tail_args = self._tail_args()
        if self._engine == 'galsim':
            script_05_galsim.main(self.args)
        elif self._engine == 'romanisim':
            script_05_romanisim.main(self.args)
            script_calculate_snrs.main(tail_args)
        else:
            raise ValueError(f"Unsupported imaging engine: {self._engine!r}")
        script_06_h5_export.main(tail_args)

    def _tail_args(self):
        """Args for the tail steps, with ``prev_step`` pointing at the exposure step.

        ``--prev-step`` is registered for _04 (which reads lens pickles from '02' or '03'),
        but calculate_snrs and _06_h5_export read the same attribute to find *exposures*.
        Handing them the _04 value sends them to an empty directory, so override it with
        whichever step this engine's exposures came from.
        """
        tail_args = copy.copy(self.args)
        if self._engine == 'galsim':
            tail_args.prev_step = script_05_galsim.SCRIPT_NAME
        else:
            tail_args.prev_step = script_05_romanisim.SCRIPT_NAME
        return tail_args

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
        '1a' : Generate galaxy tables
        '1b' : Run survey simulation (using pre-computed tables; JAX ray-shooting when jaxtronomy.use_jax)
        2 : Build lens list
        3 : Generate subhalos
        4 : Create synthetic images (JAX ray-shooting when jaxtronomy.use_jax)
        5 : Create exposures (_05_romanisim when imaging.engine is 'romanisim';
            its --level/--dither-pattern flags select the L2 or L3 variant, default L2)
        'snr' : Calculate SNRs (romanisim tail only)
        6 : Export to HDF5 file

        Raises
        ------
        ValueError
            If the script_number is not valid.
        """
        if script_number == 0:
            script_00_cache_psfs.main(self.args)
        elif script_number == '1a':
            script_01a_generate_galaxy_tables.main(self.args)
        elif script_number == '1b':
            script_01b_run_survey_simulation.main(self.args)
        elif script_number == 2:
            script_02_build_lens_list.main(self.args)
        elif script_number == 3:
            script_03_generate_subhalos.main(self.args)
        elif script_number == 4:
            script_04_create_synthetic_images.main(self.args)
        elif script_number == 5:
            if self._engine == 'romanisim':
                script_05_romanisim.main(self.args)
            else:
                script_05_galsim.main(self.args)
        elif script_number == 'snr':
            script_calculate_snrs.main(self._tail_args())
        elif script_number == 6:
            script_06_h5_export.main(self._tail_args())
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

        # copy so the local 'pipeline_label' mutation below doesn't touch self.config
        config = dict(self.config)

        if config['dev']:
            config['pipeline_label'] += '_dev'

        with open(os.path.join(config['data_dir'], config['pipeline_label'], 'execution_times.json'), 'r') as f:
            execution_times = json.load(f)

        logger.info(execution_times)

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

        logger.info(
            f'Total pipeline execution time: {datetime.timedelta(seconds=total_time)}, {total_time} seconds, {total_time / 3600:.2f} hours')
        
        data_dir = os.path.join(config['data_dir'], config['pipeline_label'], script_05_galsim.SCRIPT_NAME)

        band = config['synthetic_image']['bands'][0]
        pickles = sorted(glob(os.path.join(data_dir, 'sca*', f'Exposure_*_{band}.pkl')))
        # pickles = sorted(glob(os.path.join(data_dir, 'Exposure_*.pkl')))

        if len(pickles) > 0:
            logger.info(f'Execution time per system: {total_time / len(pickles):.3f} seconds')
        else:
            raise ValueError(f'No pickled exposures found in {data_dir}')
