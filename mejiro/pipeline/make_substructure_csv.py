import os
import re
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
config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'roman_data_challenge_rung_1.yaml')
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
        substructure_flag = None
    else:
        mf_name = realization.rendering_classes[0]._mass_function_model.__name__
        conc_name = realization.kwargs_halo_model['concentration_model_subhalos'].__class__.__name__
        match = re.match(r'^([A-Z]{2,}?)(?=[A-Z][a-z]|$)', mf_name) or re.search(r'([A-Z]{2,}?)(?=[A-Z][a-z]|$)', conc_name)
        substructure_flag = match.group(1)

    rows.append({"id": system_id, "substructure_flag": substructure_flag})

df = pd.DataFrame(rows, columns=["id", "substructure_flag"])
print(df.head())

csv_path = os.path.join(output_dir, "substructure_flags.csv")
df.to_csv(csv_path, index=False)
print(f"Wrote {len(df)} rows to {csv_path}")