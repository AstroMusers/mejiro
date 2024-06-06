import numpy as np
import csv
import os
from astropy import units as u
import speclite.filters
from tqdm import tqdm
from astropy.table import Table
from astropy.io import ascii

import slsim

_filter_name_list = [
    "F062",
    "F087",
    "F106",
    "F129",
    "F158",
    "F184",
    "F146",
    "F213",
    "Prism",
    "Grism_1stOrder",
    "Grism_0thOrder",
]


# def filter_names():
#     """

#     :return: list of filter names with full path
#     """
#     path = os.path.dirname(slsim.__file__)
#     module_path, _ = os.path.split(path)
#     save_path = os.path.join(module_path, "data/Filters/Roman/")
#     _filter_names = [
#         str(save_path + "Roman-" + name + ".ecsv") for name in _filter_name_list
#     ]
#     return _filter_names


# def average


def ecsv_to_csv(ecsv_path):
    """
    Convert ecsv file to csv file
    :param ecsv_path: path to ecsv file
    :return: None
    """
    table = ascii.read(ecsv_path)
    csv_path = ecsv_path.replace(".ecsv", ".csv")
    table.write(csv_path, format="ascii.csv", overwrite=True)
    return csv_path


def configure_roman_filters():
    import mejiro
    module_path = os.path.dirname(mejiro.__file__)
    effarea_dir = os.path.join(module_path, 'data', 'effarea')
    save_path = os.path.join(module_path, 'data', 'filter_response')

    from mejiro.utils import util
    util.create_directory_if_not_exists(save_path)
    util.clear_directory(save_path)

    from glob import glob
    effarea_files = sorted(glob(f'{effarea_dir}/*.ecsv'))

    area = (2.4 / 2) ** 2 * np.pi  # Roman mirror diameter is 2.4 meters
    # we need to divide by the area as the throughputs are given in effective area
    # in m^2

    for i, ecsv in tqdm(enumerate(effarea_files)):
        file_name_roman = ecsv_to_csv(ecsv)

        sca = i + 1
        group_name = f'RomanSCA{str(sca).zfill(2)}'

        wave = []
        responses = {
            "F062": [],
            "F087": [],
            "F106": [],
            "F129": [],
            "F158": [],
            "F184": [],
            "F146": [],
            "F213": [],
            "Prism": [],
            "Grism_1stOrder": [],
            "Grism_0thOrder": [],
        }
        with open(file_name_roman, newline="") as myFile:
            reader = csv.DictReader(myFile)
            for row in reader:
                wave.append(float(row["Wave"]))
                for name in _filter_name_list:
                    responses[name].append(float(row[name]) / area)
        wave = np.array(wave)

        # convert micrometer to Angstrom
        wavelength = wave * 10000 * u.Angstrom

        for i, filter_name in enumerate(_filter_name_list):
            response = np.array(responses[filter_name])

            # for speclite the response curve must start with 0 and end with 0
            if response[0] != 0:
                wavelength = np.insert(wavelength, 0, min(wavelength) - 100 * u.Angstrom)
                response = np.insert(response, 0, 0)
            if response[-1] != 0:
                wavelength = np.append(wavelength, max(wavelength) + 100 * u.Angstrom)
                response = np.append(response, 0)

            speclite_filter = speclite.filters.FilterResponse(
                wavelength=wavelength,
                response=response,
                meta=dict(group_name=group_name, band_name=filter_name),
            )
            speclite_filter.save(save_path)
