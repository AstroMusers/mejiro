import yaml
from glob import glob
from astropy.table import Table
from astropy.table import QTable


def return_qtable(path):
    filepath = _handle_glob(path)
    return QTable.read(filepath)


def return_table(path):
    filepath = _handle_glob(path)
    return Table.read(filepath)


def return_yaml(path):
    filepath = _handle_glob(path)
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def _handle_glob(path):
    filepaths = sorted(glob(path))

    if len(filepaths) == 0:
        raise FileNotFoundError(f"No files found matching {path}.")
    elif len(filepaths) > 1:
        raise FileExistsError(f"Multiple files found matching {path}: {filepaths}.")
    else:
        return filepaths[0]
    