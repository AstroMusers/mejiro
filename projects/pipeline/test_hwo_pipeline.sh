#!/bin/bash

config="hwo_test"

# escape if error encountered
set -e

# execute Python scripts sequentially
echo 'Running survey simulation...'
python3 01_run_survey_simulation.py --config $config
echo 'Identified detectable strong lenses.'

echo 'Building lens list from SkyPy...'
python3 02_build_lens_list.py --config $config
echo 'Built lens list.'

echo 'Adding subhalos with PyHalo...'
python3 03_generate_subhalos.py --config $config
echo 'Added subhalos.'

echo 'Building models...'
python3 04_create_synthetic_images.py --config $config
echo 'Built models.'

# echo 'Simulating images...'
# python3 05_create_exposures.py --config $config
# echo 'GalSim simulations complete.'