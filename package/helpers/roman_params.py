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
