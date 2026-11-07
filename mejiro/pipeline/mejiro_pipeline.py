import argparse
import copy
import logging
import os
import time

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
    config_schema,
)
from mejiro.pipeline.pipeline_helper import PipelineHelper, load_config

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
        # _05_romanisim flags, registered here so the dispatched Namespace is
        # complete for every script
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
        self.config = load_config(self.config_file)
        config_schema.validate(self.config, 'pipeline', source=self.config_file)

        # imaging.engine selects the exposure/export tail; step 04 reads
        # jaxtronomy.use_jax itself.
        self._engine = self.config['imaging']['engine']

    def run(self):
        """
        Executes the mejiro pipeline end-to-end.

        Runs, in order:

        1. ``_00_cache_psfs``: cache PSFs (skipped in test mode).
        2. ``_01a_generate_galaxy_tables``: draw the galaxy population tables.
        3. ``_01b_run_survey_simulation``: find detectable strong lenses.
        4. ``_02_build_lens_list``: convert them to mejiro ``StrongLens`` objects.
        5. ``_03_generate_subhalos``: add pyHalo substructure.
        6. ``_04_create_synthetic_images``: ray-shoot the synthetic images.
        7. The exposure and export steps, chosen by ``imaging.engine``:

           - ``galsim``: ``_05_galsim``, then ``_06_h5_export``.
           - ``romanisim``: ``_05_romanisim``, then ``calculate_snrs``, then ``_06_h5_export``.

        Returns
        -------
        None

        Notes
        -----
        Each step's ``main`` receives ``self.args``. The steps after ``_05_*`` receive a copy
        whose ``prev_step`` points at the exposure step (see ``_tail_args``).
        """
        self._validate_run()
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

    def _validate_run(self):
        """Check the config for every step ``run`` will execute, before the first one starts.

        Each step validates the config for itself too, but a key that only step 06 needs
        shouldn't take a full run to find out about.
        """
        PipelineHelper.resolve_data_dir(self.config, self.args.data_dir)

        scripts = [script_01a_generate_galaxy_tables, script_01b_run_survey_simulation,
                   script_02_build_lens_list, script_03_generate_subhalos,
                   script_04_create_synthetic_images]
        if not self._test_mode:
            scripts.insert(0, script_00_cache_psfs)
        steps = [(script.SCRIPT_NAME, None) for script in scripts]

        exposure_step = self._tail_args().prev_step
        steps.append((exposure_step, None))
        if self._engine == 'romanisim':
            steps.append((script_calculate_snrs.SCRIPT_NAME, exposure_step))
        steps.append((script_06_h5_export.SCRIPT_NAME, exposure_step))

        config_schema.validate_steps(self.config, steps, level=self.args.level, source=self.config_file)

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
        Run a single pipeline step.

        Parameters
        ----------
        script_number : int or str
            Which step to run. Integers for the numbered steps, strings for the others:

            - ``0``: ``_00_cache_psfs``, cache PSFs.
            - ``'1a'``: ``_01a_generate_galaxy_tables``, draw the galaxy population tables.
            - ``'1b'``: ``_01b_run_survey_simulation``, find detectable lenses (JAX
              ray-shooting when ``jaxtronomy.use_jax``).
            - ``2``: ``_02_build_lens_list``, build the lens list.
            - ``3``: ``_03_generate_subhalos``, add substructure.
            - ``4``: ``_04_create_synthetic_images``, create synthetic images (JAX
              ray-shooting when ``jaxtronomy.use_jax``).
            - ``5``: create exposures, with ``_05_romanisim`` when ``imaging.engine`` is
              ``'romanisim'`` (using its default level, L2) and ``_05_galsim`` otherwise.
            - ``'snr'``: ``calculate_snrs``, calculate SNRs (romanisim path only).
            - ``6``: ``_06_h5_export``, export to an HDF5 file.

        Raises
        ------
        ValueError
            If ``script_number`` is not one of the values above.
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
        and prints the average execution time per system based on the number of exposure files found.
        Raises
        ------
        ValueError
            If no exposures are found in the expected directory.
        Notes
        -----
        - Assumes the existence of a configuration file and an execution_times.json file.
        - Reads from the ``data_dir`` given to ``Pipeline``, or the config's if none was given.
        - The method modifies the pipeline label if running in development mode.
        """
        import datetime
        import json
        from glob import glob

        # copy so the local 'pipeline_label' mutation below doesn't touch self.config
        config = dict(self.config)

        if config['dev']:
            config['pipeline_label'] += '_dev'

        pipeline_dir = os.path.join(self.args.data_dir or config['data_dir'], config['pipeline_label'])

        with open(os.path.join(pipeline_dir, 'execution_times.json'), 'r') as f:
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

        # level 100, like util.print_execution_time, so the metrics show at any logging_level
        logger.log(
            100, f'Total pipeline execution time: {datetime.timedelta(seconds=total_time)}, {total_time} seconds, {total_time / 3600:.2f} hours')

        exposure_dir = os.path.join(pipeline_dir, script_05_galsim.SCRIPT_NAME)

        # .npz or .pkl, depending on imaging.serialization
        band = config['synthetic_image']['bands'][0]
        exposures = sorted(glob(os.path.join(exposure_dir, 'sca*', f'Exposure_*_{band}.*')))

        if len(exposures) > 0:
            logger.log(100, f'Execution time per system: {total_time / len(exposures):.3f} seconds')
        else:
            raise ValueError(f'No exposures found in {exposure_dir}')
