import os
import json
import pandas as pd

import mejiro
from mejiro.instruments.instrument_base import InstrumentBase


class Roman(InstrumentBase):

    def __init__(self):
        name = 'Roman'

        super().__init__(
            name
        )

        module_path = os.path.dirname(mejiro.__file__)
        csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
        self.df = pd.read_csv(csv_path)

        # load SCA-specific zeropoints
        self.zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

        self.pixel_scale = 0.11  # arcsec/pixel
        self.diameter = 2.4  # m

    # retrieved 25 June 2024 from https://outerspace.stsci.edu/pages/viewpage.action?spaceKey=ISWG&title=Roman+WFI+and+Observatory+Performance
    psf_fwhm = {
        'F062': 0.058,
        'F087': 0.073,
        'F106': 0.087,
        'F129': 0.106,
        'F158': 0.128,
        'F184': 0.146,
        'F213': 0.169,
        'F146': 0.105
    }

    psf_jitter = 0.012  # arcsec per axis

    def get_zeropoint(self, band, sca_id):
        """
        Return AB zeropoint in given band for the given SCA
        """
        # might provide int 1 or string 01 or string SCA01
        if type(sca_id) is int:
            sca = f'SCA{sca_id}'
        else:
            if sca_id.startswith('SCA'):
                sca = sca_id
            else:
                sca = f'SCA{sca_id}'

        return self.zp_dict[sca][band.upper()]

    def translate_band(input):
        # capitalize and remove spaces
        input = input.upper().replace(' ', '')

        # some folks are referring to Roman filters using the corresponding ground-based filters (r, z, Y, etc.)
        options_dict = {
            'F062': ['F062', 'R', 'R062'],
            'F087': ['F087', 'Z', 'Z087'],
            'F106': ['F106', 'Y', 'Y106'],
            'F129': ['F129', 'J', 'J129'],
            'F158': ['F158', 'H', 'H158'],
            'F184': ['F184', 'H/K'],
            'F146': ['F146', 'WIDE', 'W146'],
            'F213': ['F213', 'KS', 'K213']
        }

        for band, possible_names in options_dict.items():
            if input in possible_names:
                return band
            else:
                raise ValueError(f"Band {input} not recognized. Valid bands (and aliases) are {options_dict}.")

    def get_psf_fwhm(self, band):
        """
        Return PSF FWHM in given band in arcsec. Note from STScI: "PSF FWHM in arcseconds simulated for a detector near the center of the WFI FOV using an input spectrum for a K0V type star."
        """
        return self.psf_fwhm[band.upper()]

    # retrieved 31 July 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html
    thermal_backgrounds = {
        'F062': 0.003,
        'F087': 0.003,
        'F106': 0.003,
        'F129': 0.003,
        'F158': 0.048,
        'F184': 0.155,
        'F213': 4.38,
        'F146': 1.03
    }

    def get_thermal_bkg(self, band):
        """
        Return internal thermal background in given band in counts/sec/pixel
        """
        return self.thermal_backgrounds[band.upper()]

    # retrieved 31 July 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html
    min_zodi = {
        'F062': 0.25,
        'F087': 0.251,
        'F106': 0.277,
        'F129': 0.267,
        'F158': 0.244,
        'F184': 0.141,
        'F213': 0.118,
        'F146': 0.781
    }

    def get_min_zodi(self, band):
        """
        Return minimum zodiacal light level in given band in counts/pixel/sec (count rate per pixel)
        """
        return self.min_zodi[band.upper()]

    # retrieved 25 June 2024 from https://iopscience.iop.org/article/10.3847/1538-4357/aac08b/pdf
    # TODO these numbers are likely outdated, so recalculate based on latest effective area curves
    ab_zeropoints = {
        'F062': 26.99,
        'F087': 26.39,
        'F106': 26.41,
        'F129': 26.35,
        'F158': 26.41,
        'F184': 25.96
    }

    def get_ab_zeropoint(self, band):
        """
        Return AB zeropoint in given band
        """
        return self.ab_zeropoints[band.upper()]

    def get_filter_centers(self):
        roman_filters = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        fields = [f'WFI_Filter_{filter}_Center' for filter in roman_filters]

        dict = {}
        for roman_filter, field in zip(roman_filters, fields):
            dict[roman_filter] = float(self.df.loc[self.df['Name'] == field]['Value'].to_string(index=False))

        return dict

    def get_pixel_scale(self):
        return float(self.df.loc[self.df['Name'] == 'WFI_Pixel_Scale']['Value'].to_string(index=False))

    def get_min_max_wavelength(self, band):
        range = self.df.loc[self.df['Name'] == f'WFI_Filter_{band.upper()}_Wavelength_Range']['Value'].to_string(
            index=False)
        min, max = range.split('-')
        return float(min), float(max)

    def get_min_zodi_count_rate(self, band):
        count_rate = self.df.loc[self.df['Name'] == f'WFI_Count_Rate_Zody_Minimum_{band.upper()}']['Value'].to_string(
            index=False)
        return float(count_rate)
