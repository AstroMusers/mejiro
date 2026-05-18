import h5py
import numpy as np
from tqdm.auto import tqdm

label = 'roman_data_challenge_rung_0'
version = 'v_1_2'
h5_filepath = f'/data/bwedig/mejiro/{label}/06/{label}_{version}.h5'

with h5py.File(h5_filepath, 'r+') as f:
    images = f['images']
    updated = 0
    for lens_name in tqdm(images):
        lens_group = images[lens_name]
        for dataset_name in lens_group:
            dataset = lens_group[dataset_name]
            if 'units' in dataset.attrs:
                current = dataset.attrs['units']
                if current[0] == 'Counts/sec':
                    new_value = np.array(['Counts', current[1]], dtype=current.dtype)
                    del dataset.attrs['units']
                    dataset.attrs['units'] = new_value
                    updated += 1

print(f"Updated 'units' attribute from 'Counts/sec' to 'Counts' for {updated} exposure datasets.")
