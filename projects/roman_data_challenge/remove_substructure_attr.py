import h5py

h5_filepath = '/data/bwedig/mejiro/roman_data_challenge_rung_1_unlabeled/06/roman_data_challenge_rung_1_unlabeled_v_0_1.h5'

with h5py.File(h5_filepath, 'r+') as f:
    images = f['images']
    removed = 0
    for name in images:
        group = images[name]
        if 'substructure' in group.attrs:
            del group.attrs['substructure']
            removed += 1

print(f"Removed 'substructure' attribute from {removed} strong lens groups.")
