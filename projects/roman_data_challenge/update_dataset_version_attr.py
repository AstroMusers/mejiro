import h5py

label = 'roman_data_challenge_rung_1'  # _unlabeled
version = 'v_2_0'
h5_filepath = f'/data/bwedig/mejiro/{label}/06/{label}_{version}.h5'

new_dataset_version = 2.0

with h5py.File(h5_filepath, 'r+') as f:
    old_dataset_version = f.attrs['dataset_version']
    f.attrs['dataset_version'] = new_dataset_version

print(f"Updated 'dataset_version' attribute from '{old_dataset_version}' to '{new_dataset_version}'.")
