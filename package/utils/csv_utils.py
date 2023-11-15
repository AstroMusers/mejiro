import os
from csv import DictReader, DictWriter

import pandas as pd

from package.utils import utils


def dataset_list_to_csv(dataset_list, csv_filepath):
    dict_list = []

    for dataset in dataset_list:
        dict_list.append(dataset.__dict__)
    
    dict_list_to_csv(dict_list, csv_filepath)


def dict_list_to_csv(dict_list, csv_filepath):
    if dict_list is not None:
        keys = utils.get_dict_keys_as_list(dict_list[0])

        with open(csv_filepath, 'w') as csv_file:
            writer = DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dict_list)
    else:
        raise Exception('Dictionary list is empty')


def csv_to_dict_list(csv_filepath):
    with open(csv_filepath, mode='r', encoding='utf-8-sig') as f:
        dict_reader = DictReader(f)
        list_to_return = list(dict_reader)
     
    return list_to_return


def combine_all_csvs(path, filename):
    # list all files in directory
    csv_files = [f for f in os.listdir(path) if not f.startswith('.')]

    # concatenate CSVs
    pd_list = []

    for f in csv_files:
        pd_list.append(pd.read_csv(os.path.join(path, f)))

    df_res = pd.concat(pd_list, ignore_index=True)

    df_res.to_csv(filename)


def remove_bom(filepath):
    s = open(filepath, mode='r', encoding='utf-8-sig').read()
    open(filepath, mode='w', encoding='utf-8').write(s)
