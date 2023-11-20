import os
from datetime import date


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dict_keys_as_list(dict):
    list = []
    for key in dict.keys():
        list.append(key)

    return list


def get_today_str():
    return str(date.today())
