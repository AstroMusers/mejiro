import os
import yaml
from astropy.table import Table
from astropy.table import QTable

import mejiro
from mejiro.instruments.instrument import Instrument
from mejiro.utils import roman_util, rti_util


# paths
ZEROPOINT_PATH = 'data/WideFieldInstrument/Imaging/ZeroPoints/Roman_zeropoints_*.ecsv'
ZODIACAL_LIGHT_PATH = 'data/WideFieldInstrument/Imaging/ZodiacalLight/zodiacal_light.ecsv'
THERMAL_BACKGROUND_PATH = 'data/WideFieldInstrument/Imaging/Backgrounds/internal_thermal_backgrounds.ecsv'


class Roman(Instrument):
    def __init__(self, roman_technical_information_path):
        name = 'Roman'
        bands = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        engines = ['galsim', 'lenstronomy', 'pandeia', 'romanisim']

        # read from roman-technical-documentation
        # module_path = os.path.dirname(mejiro.__file__)
        # rti_path = os.path.join(module_path, 'data', 'roman-technical-documentation')

        # record version of roman-technical-documentation
        try:
            version_path = os.path.join(roman_technical_information_path, 'VERSION.md')
            with open(version_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Check if the file has at least two lines
            if len(lines) >= 2:
                version_number = lines[1].strip()
            else:
                raise IndexError("Error reading version number from VERSION.md: not enough lines.")
        except FileNotFoundError:
            raise FileNotFoundError(f"VERSION.md not found in {roman_technical_information_path}.")

        super().__init__(
            name,
            bands,
            engines
        )

        self.version = version_number

        # fields to initialize: these can be retrieved on demand
        self.zeropoints = None
        self.thermal_background = None
        self.minimum_zodiacal_light = None
        self.psf_fwhm = None
    
    def get_pixel_scale(self, band=None):
        return 0.11
    
    def get_zeropoint_magnitude(self, band):
        if self.zeropoints is None:
            self.zeropoints = rti_util.return_qtable(ZEROPOINT_PATH)

        
    def get_psf_fwhm(self, band):
        # TODO TEMP
        pass

    def validate_instrument_params(self, params):
        # TODO TEMP
        pass

    def default_params(self):
        # TODO TEMP
        pass
