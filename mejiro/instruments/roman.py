import os
import warnings
import yaml
from astropy.table import Table
from astropy.table import QTable
from astropy.units import Quantity

import mejiro
from mejiro.instruments.instrument import Instrument
from mejiro.utils import roman_util, rti_util


ZEROPOINT_PATH = 'data/WideFieldInstrument/Imaging/ZeroPoints/Roman_zeropoints_*.ecsv'
ZODIACAL_LIGHT_PATH = 'data/WideFieldInstrument/Imaging/ZodiacalLight/zodiacal_light.ecsv'
THERMAL_BACKGROUND_PATH = 'data/WideFieldInstrument/Imaging/Backgrounds/internal_thermal_backgrounds.ecsv'
FILTER_PARAMS_PATH = 'data/WideFieldInstrument/Imaging/FiltersSummary/filter_parameters.ecsv'


class Roman(Instrument):
    def __init__(self, roman_technical_information_path):
        name = 'Roman'
        bands = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        engines = ['galsim', 'lenstronomy', 'pandeia', 'romanisim']

        super().__init__(
            name,
            bands,
            engines
        )

        self.pixel_scale = Quantity(0.11, 'arcsec / pix')
        self.roman_technical_information_path = roman_technical_information_path

        # read from roman-technical-documentation
        # module_path = os.path.dirname(mejiro.__file__)
        # rti_path = os.path.join(module_path, 'data', 'roman-technical-documentation')

        # record version of roman-technical-documentation
        try:
            version_path = os.path.join(roman_technical_information_path, 'VERSION.md')
            with open(version_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            if len(lines) >= 2:
                self.version = lines[1].strip()
            else:
                raise IndexError("Error reading version number from VERSION.md: not enough lines.")
        except FileNotFoundError:
            raise FileNotFoundError(f"VERSION.md not found in {roman_technical_information_path}.")

        # fields to initialize: these can be retrieved on demand
        self.zeropoints = None
        self.thermal_background = None
        self.minimum_zodiacal_light = None
        self.psf_fwhm = None

    def get_pixel_scale(self, band=None):
        return self.pixel_scale
    
    def get_zeropoint_magnitude(self, band, detector):
        sca_number = roman_util.get_sca_int(detector)
        if self.zeropoints is None:
            self.zeropoints = rti_util.return_qtable(os.path.join(self.roman_technical_information_path, ZEROPOINT_PATH))
        return self.zeropoints[(self.zeropoints['element'] == band) & (self.zeropoints['detector'] == f'WFI{str(sca_number).zfill(2)}')]['ABMag']
        
    def get_thermal_background(self, band):
        if self.thermal_background is None:
            self.thermal_background = rti_util.return_qtable(os.path.join(self.roman_technical_information_path, THERMAL_BACKGROUND_PATH))
        return self.thermal_background[self.thermal_background['filter'] == band]['rate']
        
    def get_minimum_zodiacal_light(self, band):
        if self.minimum_zodiacal_light is None:
            self.minimum_zodiacal_light = rti_util.return_qtable(os.path.join(self.roman_technical_information_path, ZODIACAL_LIGHT_PATH))
        return self.minimum_zodiacal_light[self.minimum_zodiacal_light['filter'] == band]['rate']
        
    def get_psf_fwhm(self, band):
        if self.psf_fwhm is None:
            self.psf_fwhm = rti_util.return_qtable(os.path.join(self.roman_technical_information_path, FILTER_PARAMS_PATH))
        return self.psf_fwhm[self.psf_fwhm['filter'] == band]['PSF_FWHM']

    @staticmethod
    def default_params():
        return {
            'detector': 1,
            'detector_position': (2048, 2048)
        }
    
    @staticmethod
    def validate_instrument_params(params):
        if 'detector' in params:
            detector = roman_util.get_sca_int(params['detector'])
            assert detector in range(1, 19), 'Detector number must be an integer between 1 and 18.'
        else:
            default_detector = Roman.default_params()['detector']
            warnings.warn(f'No detector number provided. Defaulting to detector {default_detector}.')
            params['detector'] = default_detector

        if 'detector_position' in params:
            assert isinstance(params['detector_position'], tuple) and len(params['detector_position']) == 2 and all(
                isinstance(x, int) for x in
                params['detector_position']), 'The detector_position parameter must be an (x,y) coordinate tuple.'
            assert params['detector_position'][0] in range(4, 4092 + 1) and params['detector_position'][1] in range(4,
                                                                                                                    4092 + 1), 'Choose a valid pixel position on the range 4-4092.'
        else:
            default_detector_position = Roman.default_params()['detector_position']
            warnings.warn(f'No detector position provided. Defaulting to {default_detector_position}')
            params['detector_position'] = default_detector_position

        return params
