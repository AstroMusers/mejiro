#!/bin/bash

# make sure appropriate conda env is active
# conda activate mejiro

# shellcheck disable=SC2164
cd scripts/execute_pipeline

# execute Python scripts sequentially
echo 'Building lens list from SkyPy...'
python3 01_lens_list_from_skypy.py
echo 'Built lens list.'

echo 'Adding subhalos with PyHalo...'
python3 02_add_subhalos.py
echo 'Added subhalos.'

echo 'Building models...'
python3 03_build_models.py
echo 'Built models.'

echo 'Simulating Pandeia images...'
python3 -W ignore 04_pandeia.py  # suppress warnings
echo 'Pandeia simulations complete.'