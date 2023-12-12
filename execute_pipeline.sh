#!/bin/bash

# make sure appropriate conda env is active
# conda activate pandeia

# execute Python scripts sequentially
echo 'Building lens list from SkyPy...'
python3 01_lens_list_from_skypy.py
echo 'Built lens list.'

echo 'Adding subhalos with PyHalo...'
python3 02_add_subhalos.py
echo 'Added subhalos.'

# python3 03_build_models.py