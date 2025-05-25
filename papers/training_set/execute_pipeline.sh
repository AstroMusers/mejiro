#!/bin/bash

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

# escape if error encountered
set -e

# execute Python scripts sequentially
echo 'Survey simulation...'
python3 01_run_survey_simulation.py
echo 'Identified detectable strong lenses.'

echo 'Building lens list from SkyPy...'
python3 02_build_lens_list.py
echo 'Built lens list.'

# echo 'Adding subhalos with PyHalo...'
# python3 03_generate_subhalos.py
# echo 'Added subhalos.'

echo 'Building models...'
python3 04_create_synthetic_images.py
echo 'Built models.'

echo 'Simulating images...'
python3 05_create_exposures.py
echo 'GalSim simulations complete.'

# echo 'Generating h5 file...'
# python3 06_h5_export.py
# echo 'h5 file generation complete.'