import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from collections import Counter
from glob import glob
from pprint import pprint
from tqdm import tqdm

from mejiro.utils import util

# read configuration file
import mejiro
config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'jwst.yaml')
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

if config['dev']:
    config['pipeline_label'] += '_dev'

data_dir = os.path.join(config['data_dir'], config['pipeline_label'], '02')

pickles = sorted(glob(os.path.join(data_dir, '*.pkl')))
lenses = [util.unpickle(f) for f in pickles]
print(f'Found {len(lenses)} system(s) in {data_dir}')

galaxy_ids = [l.physical_params.get('galaxy_id') for l in lenses]
repeat_list = []

counts = Counter(galaxy_ids)
for galaxy_id, count in counts.items():
    # print(f'galaxy_id {galaxy_id} appears {count} times')
    repeat_list.append(count)

pickle_output_dir = os.path.join(config['data_dir'], config['pipeline_label'], 'pickles')
util.create_directory_if_not_exists(pickle_output_dir)
util.clear_directory(pickle_output_dir)

for l in tqdm(lenses):
    lens_json = {
        'kwargs_model': l.kwargs_model,
        'physical_params': l.physical_params,
        'kwargs_params': l.kwargs_params
    }

    pkl_fname = os.path.join(pickle_output_dir, f'{l.name}.pkl')
    with open(pkl_fname, 'wb') as pf:
        pickle.dump(lens_json, pf, protocol=pickle.HIGHEST_PROTOCOL)
