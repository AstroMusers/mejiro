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
