import pandas as pd

'''
CSV of Roman spacecraft and instrument parameters found at https://roman.ipac.caltech.edu/sims/Param_db.html. As of 2023-11-30, Phase C (released 2023-01-20) is current.
'''


class RomanParameters:
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

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

    def get_psf_fwhm(self, band):
        """
        Return PSF FWHM in given band in arcsec. Note from STScI: "PSF FWHM in arcseconds simulated for a detector near the center of the WFI FOV using an input spectrum for a K0V type star."
        """
        return self.psf_fwhm[band.upper()]

    # retrieved 25 June 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html
    thermal_backgrounds = {
        'F062': 0.,
        'F087': 0.,
        'F106': 0.,
        'F129': 0.,
        'F158': 0.04,
        'F184': 0.17,
        'F213': 4.52,
        'F146': 0.98
    }

    def get_thermal_bkg(self, band):
        """
        Return internal thermal background in given band in counts/sec/pixel
        """
        return self.thermal_backgrounds[band.upper()]

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
