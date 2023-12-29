import datetime
import os
import pickle as _pickle
from collections import ChainMap
import shutil

from glob import glob
from omegaconf import OmegaConf


def center_crop_image(array, shape):
    y_out, x_out = shape
    y, x = array.shape
    x_start = (x // 2) - (x_out // 2)
    y_start = (y // 2) - (y_out // 2)    
    return array[y_start:y_start+y_out, x_start:x_start+x_out]


def hydra_to_dict(config):
    container = OmegaConf.to_container(config, resolve=True)
    return dict(ChainMap(*container))


def print_execution_time(start, stop):
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def pickle(path, object):
    with open(path, 'ab') as results_file:
        _pickle.dump(object, results_file)


def unpickle(path):
    with open(path, 'rb') as results_file:
        result = _pickle.load(results_file)
    return result


def unpickle_all(dir_path, prefix='', limit=None):
    file_list = glob(dir_path + f'/{prefix}*')
    if limit is not None:
        return [unpickle(i) for i in file_list[:limit] if os.path.isfile(i)]
    else:
        return [unpickle(i) for i in file_list if os.path.isfile(i)]


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_directory(path):
    for i in glob(path + '/*'):
        if os.path.isfile(i):
            os.remove(i)
        else:
            shutil.rmtree(i)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def scientific_notation_string(input):
    return '{:.2e}'.format(input)


def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


# TODO finish
# def scientific_notation_string(input):
#     # convert to Python scientific notion
#     string = '{:e}'.format(input)
#     num_string, exponent = string.split('e')
#     num = str(round(float(num_string), 2))

#     # handle exponent
#     if exponent[0] == '+':
#         _, power = exponent.split('+')
#     elif exponent[0] == '-':
#         _, power = exponent.split('-')
#         power = '-' + power
        

#     power = str(int(power))
#     exponent = '10^{' + power + '}'

#     return ''.join((num, '\cross', exponent))
