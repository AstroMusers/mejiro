import importlib
import os
from glob import glob
from tqdm import tqdm

from mejiro.galaxy_galaxy import GalaxyGalaxy
from mejiro.utils import util


class PipelineHelper:
    def __init__(self, config, prev_script_name, script_name):
        self.config = config
        self.prev_script_name = prev_script_name
        self.script_name = script_name

        # get attributes from config
        self.dev = config['dev']
        self.verbose = config['verbose']
        self.data_dir = config['data_dir']
        self.limit = config['limit']
        self.detectors = config['survey']['detectors']
        self.psf_cache_dir = os.path.join(self.data_dir, config['psf_cache_dir'])

        # for key, value in self.config.items():
        #     if isinstance(value, dict):
        #         value = PipelineHelper(value)
        #     setattr(self, key, value)

        # set pipeline name
        self.name = config['pipeline_label']

        # load instrument
        self.instrument_name = config['instrument'].lower()
        self.instrument = self.initialize_instrument_class()

        # set up top directory for all pipeline output
        self.pipeline_dir = os.path.join(self.data_dir, self.config['pipeline_label'])
        if self.dev:
            self.pipeline_dir += '_dev'

        # set up input directory for current script
        self.input_dir = os.path.join(self.pipeline_dir, self.prev_script_name)

        # set up output directory for current script
        self.output_dir = os.path.join(self.pipeline_dir, self.script_name)
        util.create_directory_if_not_exists(self.output_dir)
        util.clear_directory(self.output_dir)

    def retrieve_roman_sca_input(self):
        # get input directories
        input_sca_dirs = [os.path.basename(d) for d in glob(os.path.join(self.input_dir, 'sca*')) if os.path.isdir(d)]
        if self.verbose: print(f'Reading from {input_sca_dirs}')

        # parse scas from input directories
        scas = sorted([int(d[3:]) for d in input_sca_dirs])
        scas = [str(sca).zfill(2) for sca in scas]

        return input_sca_dirs, scas

    def create_roman_sca_output_directories(self):
        output_sca_dirs = []
        for sca in self.detectors:
            sca_dir = os.path.join(self.output_dir, f'sca{str(sca).zfill(2)}')
            os.makedirs(sca_dir, exist_ok=True)
            output_sca_dirs.append(sca_dir)
        if self.verbose: print(f'Set up output directories {output_sca_dirs}')
        return output_sca_dirs

    def retrieve_roman_pickles(self, prefix, suffix, extension):
        filename_pattern = f'{prefix}_*'
        if suffix:
            filename_pattern += f'_{suffix}'
        filename_pattern += f'{extension}'
        return sorted(glob(os.path.join(self.input_dir, 'sca*', filename_pattern)))

    def retrieve_hwo_pickles(self, prefix='', suffix='', extension='.pkl'):
        filename_pattern = f'{prefix}_*'
        if suffix:
            filename_pattern += f'_{suffix}'
        filename_pattern += f'{extension}'
        return sorted(glob(os.path.join(self.input_dir, filename_pattern)))
    
    def initialize_instrument_class(self):
        base_module_path = "mejiro.instruments"
        class_map = {
            "hwo": "HWO",
            "roman": "Roman"
        }

        if self.instrument_name.lower() not in class_map:
            raise ValueError(f"Unknown instrument: {self.instrument_name}")

        module_path = f"{base_module_path}.{self.instrument_name.lower()}"
        module = importlib.import_module(module_path)
        class_name = class_map[self.instrument_name.lower()]
        cls = getattr(module, class_name)
        return cls()
