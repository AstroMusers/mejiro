#!/bin/bash

# make sure appropriate conda env is active
# conda activate mejiro

# shellcheck disable=SC2164
cd scripts/execute_pipeline

# execute Python scripts sequentially
# echo 'Checking installation...'
# python3 00_pipeline_setup.py
# echo 'Installation check passed.'

echo 'Building lens list from SkyPy...'
python3 01_lens_list_from_skypy.py
echo 'Built lens list.'

echo 'Adding subhalos with PyHalo...'
python3 02_add_subhalos.py
echo 'Added subhalos.'

echo 'Building models...'
python3 03_build_models.py
echo 'Built models.'

echo 'Simulating images...'
python3 04_galsim.py
echo 'GalSim simulations complete.'

echo 'Generating color images...'
python3 05_color.py
echo 'Color image generation complete.'