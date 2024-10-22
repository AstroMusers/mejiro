#!/bin/bash

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

# escape if error encountered
set -e

# execute Python scripts sequentially
# echo 'Simulating HLWAS to find detectable strong lenses...'
# python3 pipeline/updated_scripts/00_run_hlwas_simulation.py
# echo 'Identified detectable strong lenses.'

# echo 'Building lens list from SkyPy...'
# python3 pipeline/scripts/01_build_lens_list.py
# echo 'Built lens list.'

# echo 'Adding subhalos with PyHalo...'
# python3 pipeline/updated_scripts/02_generate_subhalos.py
# echo 'Added subhalos.'

echo 'Building models...'
python3 pipeline/updated_scripts/03_create_synthetic_images.py
echo 'Built models.'

echo 'Simulating images...'
python3 pipeline/updated_scripts/04_create_exposures.py
echo 'GalSim simulations complete.'

echo 'Generating color images...'
python3 pipeline/updated_scripts/05_make_rgb_exposures.py
echo 'Color image generation complete.'

# echo 'Generating hdf5 file...'
# python3 06_output.py
# echo 'hdf5 file generation complete.'