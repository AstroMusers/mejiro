import os
import yaml
from glob import glob
from pprint import pprint
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from mejiro.analysis import lensing
from mejiro.plots import corner
from mejiro.utils import util

# read configuration file
import mejiro
import pickle
import pandas as pd
repo_root = os.path.dirname(os.path.dirname(mejiro.__file__))
config_file = os.path.join(repo_root, 'projects', 'roman_data_challenge', 'roman_data_challenge_rung_1.yaml')
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

if config['dev']:
    config['pipeline_label'] += '_dev'

input_dir = os.path.join(config['data_dir'], config['pipeline_label'], '03')
output_dir = os.path.join(config['data_dir'], config['pipeline_label'])

lens_pickles = sorted(glob(os.path.join(input_dir, 'sca*', 'lens_*.pkl')))
print(f'Found {len(lens_pickles)} system(s) in {input_dir}')

# calculate and implement limit, if specified
if config['limit'] is not None:
    lens_pickles = lens_pickles[:config['limit']]
    print(f'Limiting to {len(lens_pickles)} systems')

rows = []
for path in tqdm(lens_pickles):
    lens = util.unpickle(path)
    realization = lens.realization

    system_id = lens.name

    if realization is None:
        # no substructure
        substructure_flag = None
    elif hasattr(lens, 'substructure_flag'):
        # lightweight serialization: realization stripped, flag stored at 03 time
        substructure_flag = lens.substructure_flag
    else:
        # full serialization: derive the flag from the realization object
        substructure_flag = lensing.substructure_flag(realization)

    rows.append({"id": system_id, "substructure_flag": substructure_flag})

df = pd.DataFrame(rows, columns=["id", "substructure_flag"])
print(df.head())

csv_path = os.path.join(output_dir, "substructure_flags.csv")
df.to_csv(csv_path, index=False)
print(f"Wrote {len(df)} rows to {csv_path}")