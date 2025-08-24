#!/bin/bash

config="roman_reproduce"  # training_set, roman_hlwas_medium

# escape if error encountered
set -e

# execute Python scripts sequentially
# echo 'Caching PSFs...'
# python3 00_cache_psfs.py --config $config
# echo 'Cached PSFs.'

echo 'Running survey simulation...'
python3 01_run_survey_simulation.py --config $config
echo 'Identified detectable strong lenses.'

echo 'Building lens list from SkyPy...'
python3 02_build_lens_list.py --config $config
echo 'Built lens list.'

# if [ $config != "training_set" ]; then
#     echo 'Adding subhalos with PyHalo...'
#     python3 03_generate_subhalos.py --config $config
#     echo 'Added subhalos.'
# fi

echo 'Building models...'
python3 04_create_synthetic_images.py --config $config
echo 'Built models.'

echo 'Simulating images...'
python3 05_create_exposures.py --config $config
echo 'GalSim simulations complete.'

echo 'Calculating SNRs...'
python3 calculate_snrs.py --config $config
echo 'SNR calculation complete.'

# echo 'Generating h5 file...'
# if [ $config == "training_set" ]; then
#     python3 06_h5_export_training_set.py --config $config
# else
#     python3 06_h5_export.py --config $config
# fi
# echo 'h5 file generation complete.'