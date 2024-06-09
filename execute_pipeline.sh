#!/bin/bash

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

# escape if error encountered
set -e

# shellcheck disable=SC2164
cd scripts/pipeline

# execute Python scripts sequentially
echo 'Simulating HLWAS to find detectable strong lenses...'
python3 00_hlwas_sim.py
echo 'Identified detectable strong lenses.'

echo 'Building lens list from SkyPy...'
python3 01_build_lens_list.py
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

# echo 'Generating hdf5 file...'
# python3 06_output.py
# echo 'hdf5 file generation complete.'