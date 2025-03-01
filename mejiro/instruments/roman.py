import os
import warnings
from astropy.units import Quantity

from mejiro.instruments.instrument import Instrument
from mejiro.utils import roman_util, util


ZEROPOINT_PATH = 'data/WideFieldInstrument/Imaging/ZeroPoints/Roman_zeropoints_*.ecsv'
ZODIACAL_LIGHT_PATH = 'data/WideFieldInstrument/Imaging/ZodiacalLight/zodiacal_light.ecsv'
THERMAL_BACKGROUND_PATH = 'data/WideFieldInstrument/Imaging/Backgrounds/internal_thermal_backgrounds.ecsv'
FILTER_PARAMS_PATH = 'data/WideFieldInstrument/Imaging/FiltersSummary/filter_parameters.ecsv'


class Roman(Instrument):
    """
    Roman Space Telescope (Roman) Wide Field Instrument (WFI) class

    Attributes
    ----------
    roman_technical_information_path : str
        Path to the roman-technical-information directory.
    pixel_scale : Quantity
        Pixel scale of the instrument in arcsec/pix.
    gain : float
        Gain of the instrument.
    stray_light_fraction : float
        Fraction of stray light.
    zeropoints : QTable or None
        Zeropoints table, initialized on demand.
    thermal_background : QTable or None
        Thermal background table, initialized on demand.
    minimum_zodiacal_light : QTable or None
        Minimum zodiacal light table, initialized on demand.
    psf_fwhm : QTable or None
        PSF FWHM table, initialized on demand.
    """
    def __init__(self):
        name = 'Roman'
        bands = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        engines = ['galsim', 'lenstronomy', 'pandeia', 'romanisim']

        super().__init__(
            name,
            bands,
            engines
        )        

        # get path to roman-technical-information from environment variable
        self.roman_technical_information_path = os.getenv("ROMAN_TECHNICAL_INFORMATION_PATH")
        if self.roman_technical_information_path is None:
            raise EnvironmentError("Environment variable 'ROMAN_TECHNICAL_INFORMATION_PATH' is not set.")

        # record version of roman-technical-documentation
        try:
            version_path = os.path.join(self.roman_technical_information_path, 'VERSION.md')
            with open(version_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            if len(lines) >= 2:
                self.versions['roman-technical-information'] = lines[1].strip()
            else:
                raise IndexError("Error reading version number from VERSION.md: not enough lines.")
        except FileNotFoundError:
            raise FileNotFoundError(f"VERSION.md not found in {self.roman_technical_information_path}.")
        
        # set attributes
        self.pixel_scale = Quantity(0.11, 'arcsec / pix')
        self.gain = 1.
        self.stray_light_fraction = 0.1

        # fields to initialize: these can be retrieved on demand
        self.zeropoints = None
        self.thermal_background = None
        self.minimum_zodiacal_light = None
        self.psf_fwhm = None
    
    # implement abstract methods
    def get_minimum_zodiacal_light(self, band):
        if self.minimum_zodiacal_light is None:
            self.minimum_zodiacal_light = util.return_qtable(os.path.join(self.roman_technical_information_path, ZODIACAL_LIGHT_PATH))
        return self.minimum_zodiacal_light[self.minimum_zodiacal_light['filter'] == band]['rate']
    
    def get_pixel_scale(self, band=None):
        return self.pixel_scale
        
    def get_psf_fwhm(self, band):
        if self.psf_fwhm is None:
            self.psf_fwhm = util.return_qtable(os.path.join(self.roman_technical_information_path, FILTER_PARAMS_PATH))
        return self.psf_fwhm[self.psf_fwhm['filter'] == band]['PSF_FWHM']
    
    def get_thermal_background(self, band):
        if self.thermal_background is None:
            self.thermal_background = util.return_qtable(os.path.join(self.roman_technical_information_path, THERMAL_BACKGROUND_PATH))
        return self.thermal_background[self.thermal_background['filter'] == band]['rate']
    
    def get_zeropoint_magnitude(self, band, detector):
        sca_number = roman_util.get_sca_int(detector)
        if self.zeropoints is None:
            self.zeropoints = util.return_qtable(os.path.join(self.roman_technical_information_path, ZEROPOINT_PATH))
        return self.zeropoints[(self.zeropoints['element'] == band) & (self.zeropoints['detector'] == f'WFI{str(sca_number).zfill(2)}')]['ABMag']

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
            if detector not in range(1, 19):
                raise ValueError('Detector number must be an integer between 1 and 18.')
        else:
            default_detector = Roman.default_params()['detector']
            warnings.warn(f'No detector number provided. Defaulting to detector {default_detector}.')
            params['detector'] = default_detector

        if 'detector_position' in params:
            if not (isinstance(params['detector_position'], tuple) and len(params['detector_position']) == 2 and all(isinstance(x, int) for x in params['detector_position'])):
                    raise ValueError('The detector_position parameter must be an (x,y) coordinate tuple.')
            if not (params['detector_position'][0] in range(4, 4092 + 1) and params['detector_position'][1] in range(4, 4092 + 1)):
                    raise ValueError('Choose a valid pixel position on the range 4-4092.')
        else:
            default_detector_position = Roman.default_params()['detector_position']
            warnings.warn(f'No detector position provided. Defaulting to {default_detector_position}')
            params['detector_position'] = default_detector_position

        return params
