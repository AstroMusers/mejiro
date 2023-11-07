import csv
import os
from datetime import date

from models.dataset import Dataset


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dict_keys_as_list(dict):
    list = []
    for key in dict.keys():
        list.append(key)

    return list


def deserialize(filename):
    dataset_list = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            dataset_list.append(Dataset(data_set_name=row[0], 
                                        target_name=row[1], 
                                        ra=row[2], 
                                        dec=row[3], 
                                        start_time=row[5], 
                                        stop_time=row[6],
                                        actual_duration=row[7],
                                        instrument=row[8], 
                                        aper=row[9], 
                                        spec=row[10], 
                                        central_wavelength=row[11], release_date=row[14], 
                                        preview_name=row[15], 
                                        filename=None, 
                                        data_uri=None, 
                                        cutout=None, 
                                        peak_value=None,
                                        fits_filepath=None,
                                        cutout_filepath=None))
    f.close()

    return dataset_list


def all_equal(iterator):
    return len(set(iterator)) <= 1


def get_today_str():
    return str(date.today())
