import importlib
import os
import yaml
from glob import glob
from logging import config

import mejiro
from mejiro.utils import util


class PipelineHelper:
    def __init__(self, args, prev_script_name, script_name, supported_instruments):
        self.prev_script_name = prev_script_name
        self.script_name = script_name
        self.supported_instruments = supported_instruments

        # ensure the configuration file has a .yaml or .yml extension
        if not args.config.endswith(('.yaml', '.yml')):
            if os.path.exists(args.config + '.yaml'):
                args.config += '.yaml'
            elif os.path.exists(args.config + '.yml'):
                args.config += '.yml'
            else:
                raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")

        # read configuration file
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config

        # set data directory
        self.data_dir = config['data_dir']
        if hasattr(args, 'data_dir') and args.data_dir is not None:
            print(f'Overriding data_dir in config file ({self.data_dir}) with provided data_dir ({args.data_dir})')  # TODO logging
            self.data_dir = args.data_dir
        elif self.data_dir is None:
            raise ValueError("data_dir must be specified either in the config file or via the --data_dir argument.")
        
        # get attributes from config
        self.dev = config['dev']
        self.verbose = config['verbose']
        self.limit = config['limit']
        self.runs = config['survey']['runs']
        self.detectors = config['survey']['detectors']

        # set pipeline name
        self.name = config['pipeline_label']

        # set nice level
        os.nice(config.get('nice', 0))

        # suppress warnings
        if config['suppress_warnings']:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)

        # load instrument
        self.instrument_name = config['instrument'].lower()
        if self.instrument_name not in self.supported_instruments:
            raise ValueError(f"Unsupported instrument: {self.instrument_name}. Supported instruments are {self.supported_instruments}.")
        self.instrument = self.initialize_instrument_class()

        # set psf cache directory
        self.psf_cache_dir = config['psf_cache_dir']
        if self.psf_cache_dir is None:
            self.psf_cache_dir = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'psfs', self.instrument_name.lower())
        else:
            self.psf_cache_dir = os.path.join(self.data_dir, self.psf_cache_dir)

        # set up top directory for all pipeline output
        self.pipeline_dir = os.path.join(self.data_dir, self.config['pipeline_label'])
        if self.dev:
            self.pipeline_dir += '_dev'

        # set up input directory for current script
        if self.prev_script_name is not None:
            self.input_dir = os.path.join(self.pipeline_dir, self.prev_script_name)

        # set up output directory for current script
        if self.script_name is None:
            raise ValueError("script_name must be specified.")
        self.output_dir = os.path.join(self.pipeline_dir, self.script_name)
        util.create_directory_if_not_exists(self.output_dir)
        util.clear_directory(self.output_dir)

    def calculate_process_count(self, count):
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        process_count = self.config['cores'][f'script_{self.script_name}']
        if count < process_count:
            process_count = count
        print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    def retrieve_roman_sca_input(self):
        self.validate_instrument('roman')

        # get input directories
        input_sca_dirs = [os.path.basename(d) for d in glob(os.path.join(self.input_dir, 'sca*')) if os.path.isdir(d)]
        if self.verbose: print(f'Reading from {input_sca_dirs}')

        # parse scas from input directories
        scas = sorted([int(d[3:]) for d in input_sca_dirs])
        scas = [str(sca).zfill(2) for sca in scas]

        return input_sca_dirs, scas

    def parse_sca_from_filename(self, filename):
        self.validate_instrument('roman')

        # extract SCA from filename
        dirname = os.path.dirname(filename)
        sca = dirname.split('/')[-1]
        if sca.startswith('sca'):
            return int(sca[3:])
        else:
            raise ValueError(f'Invalid SCA format in filename: {filename}')

    def create_roman_sca_output_directories(self):
        self.validate_instrument('roman')

        # for a case where e.g. only 2 runs but 18 detectors, only create 2 folders
        detectors_to_use = self.detectors
        if self.runs < len(detectors_to_use):
            detectors_to_use = detectors_to_use[:self.runs]

        output_sca_dirs = []
        for sca in detectors_to_use:
            sca_dir = os.path.join(self.output_dir, f'sca{str(sca).zfill(2)}')
            os.makedirs(sca_dir, exist_ok=True)
            output_sca_dirs.append(sca_dir)
        if self.verbose: print(f'Set up output directories {output_sca_dirs}')
        return output_sca_dirs
    
    def parse_roman_uids(self, prefix, suffix, extension):
        uids = set()

        roman_pickles = self.retrieve_roman_pickles(prefix=prefix, suffix=suffix, extension=extension)
        for f in roman_pickles:
            basename = os.path.basename(f)
            uid = basename.split("_")[-2]
            uids.add(uid)

        return sorted(uids)

    def retrieve_roman_pickles(self, prefix, suffix, extension):
        self.validate_instrument('roman')

        filename_pattern = f'{prefix}_{self.name}_*'
        if suffix:
            filename_pattern += f'_{suffix}'
        filename_pattern += f'{extension}'
        return sorted(glob(os.path.join(self.input_dir, 'sca*', filename_pattern)))

    def retrieve_hwo_pickles(self, prefix='', suffix='', extension='.pkl'):
        self.validate_instrument('hwo')

        filename_pattern = f'{prefix}_{self.name}_*'
        if suffix:
            filename_pattern += f'_{suffix}'
        filename_pattern += f'{extension}'
        return sorted(glob(os.path.join(self.input_dir, filename_pattern)))
    
    def initialize_instrument_class(self):
        base_module_path = "mejiro.instruments"
        class_map = {
            "hwo": "HWO",
            "jwst": "JWST",
            "roman": "Roman"
        }

        if self.instrument_name.lower() not in class_map:
            raise ValueError(f"Unknown instrument: {self.instrument_name}")

        module_path = f"{base_module_path}.{self.instrument_name.lower()}"
        module = importlib.import_module(module_path)
        class_name = class_map[self.instrument_name.lower()]
        cls = getattr(module, class_name)
        return cls()

    def validate_instrument(self, instrument_name):
        assert self.instrument_name == instrument_name, f"This method is only for the {instrument_name} instrument."
