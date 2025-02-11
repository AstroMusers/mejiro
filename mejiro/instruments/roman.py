import json
import os
import pandas as pd
import warnings

import mejiro
from mejiro.instruments.instrument_base import InstrumentBase
from mejiro.utils import roman_util


class Roman(InstrumentBase):

    def __init__(self):
        name = 'Roman'
        bands = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        engines = ['galsim', 'lenstronomy', 'pandeia', 'romanisim']

        super().__init__(
            name,
            bands,
            engines
        )

        module_path = os.path.dirname(mejiro.__file__)
        csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
        self.df = pd.read_csv(csv_path)

        # load SCA-specific zeropoints
        self.zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))
        self.min_zodi_dict = json.load(open(os.path.join(module_path, 'data', 'roman_minimum_zodiacal_light.json')))
        self.thermal_bkg_dict = json.load(open(os.path.join(module_path, 'data', 'roman_thermal_background.json')))

        # ---------------------CONSTANTS---------------------
        self.pixel_scale = 0.11  # arcsec per pixel
        self.diameter = 2.4  # m
        self.psf_jitter = 0.012  # arcsec per axis
        self.pixels_per_axis = 4088
        self.total_pixels_per_axis = 4096
        self.thermal_bkg = {
            'F062': 0.003,
            'F087': 0.003,
            'F106': 0.003,
            'F129': 0.003,
            'F158': 0.048,
            'F184': 0.155,
            'F213': 4.38,
            'F146': 1.03
        }  # retrieved 31 July 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html, information dated 03 June 2024
        self.min_zodi = {
            'F062': 0.25,
            'F087': 0.251,
            'F106': 0.277,
            'F129': 0.267,
            'F158': 0.244,
            'F184': 0.141,
            'F213': 0.118,
            'F146': 0.781
        }  # retrieved 31 July 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html, information dated 03 June 2024
        self.psf_fwhm = {
            'F062': 0.058,
            'F087': 0.073,
            'F106': 0.087,
            'F129': 0.106,
            'F158': 0.128,
            'F184': 0.146,
            'F213': 0.169,
            'F146': 0.105
        }  # retrieved 25 June 2024 from https://outerspace.stsci.edu/pages/viewpage.action?spaceKey=ISWG&title=Roman+WFI+and+Observatory+Performance

    def get_pixel_scale(self, band=None):
        """
        Returns the pixel scale for Roman's WFI.

        Parameters
        ----------
        band : str, optional
            The specific band for which to get the pixel scale. For Roman's WFI, the pixel scale is the same across all bands.

        Returns
        -------
        float
            The pixel scale in arcseconds per pixel.
        """
        return self.pixel_scale

    def get_filter_centers(self):
        fields = [f'WFI_Filter_{band}_Center' for band in self.bands]

        filter_centers = {}
        for band, field in zip(self.bands, fields):
            filter_centers[band] = float(self.df.loc[self.df['Name'] == field]['Value'].to_string(index=False))

        return filter_centers

    def get_min_max_wavelength(self, band):
        range = self.df.loc[self.df['Name'] == f'WFI_Filter_{band.upper()}_Wavelength_Range']['Value'].to_string(
            index=False)
        min, max = range.split('-')
        return float(min), float(max)

    def get_min_zodi_count_rate(self, band):
        count_rate = self.df.loc[self.df['Name'] == f'WFI_Count_Rate_Zody_Minimum_{band.upper()}']['Value'].to_string(
            index=False)
        return float(count_rate)

    def get_psf_fwhm(self, band):
        """
        Return PSF FWHM in given band in arcsec. Note from STScI: "PSF FWHM in arcseconds simulated for a detector near the center of the WFI FOV using an input spectrum for a K0V type star."
        """
        return self.psf_fwhm[band.upper()]

    def get_thermal_bkg(self, band, sca):
        sca = roman_util.get_sca_string(sca)
        return self.thermal_bkg_dict[sca][band.upper()]

    def get_min_zodi(self, band, sca):
        sca = roman_util.get_sca_string(sca)
        return self.min_zodi_dict[sca][band.upper()]

    def get_zeropoint_magnitude(self, band, sca=1):
        """
        Return AB zeropoint in given band for the given SCA
        """
        sca = roman_util.get_sca_string(sca)
        return self.zp_dict[sca][band.upper()]

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
