#!/bin/bash

# this bash script is a less-polished way of executing the `mejiro` pipeline that is useful for development

config="mejiro/data/mejiro_config/roman_test.yaml"

# change directory to the root of the repository
cd "$(dirname "$0")/.."

# escape if error encountered
set -e

# execute Python scripts sequentially
# echo 'Caching PSFs...'
# python3 mejiro/pipeline/_00_cache_psfs.py --config $config
# echo 'Cached PSFs.'

echo 'Running survey simulation...'
python3 mejiro/pipeline/_01_run_survey_simulation.py --config $config
echo 'Identified detectable strong lenses.'

echo 'Building lens list from SkyPy...'
python3 mejiro/pipeline/_02_build_lens_list.py --config $config
echo 'Built lens list.'

if [ $config != "training_set" ]; then
    echo 'Adding subhalos with PyHalo...'
    python3 mejiro/pipeline/_03_generate_subhalos.py --config $config
    echo 'Added subhalos.'
fi

# echo 'Building models...'
# python3 mejiro/pipeline/_04_create_synthetic_images.py --config $config
# echo 'Built models.'

# echo 'Simulating images...'
# python3 mejiro/pipeline/_05_create_exposures.py --config $config
# echo 'GalSim simulations complete.'

# echo 'Calculating SNRs...'
# python3 mejiro/pipeline/calculate_snrs.py --config $config
# echo 'SNR calculation complete.'

# echo 'Generating h5 file...'
# if [ $config == "training_set" ]; then
#     python3 mejiro/pipeline/_06_h5_export_training_set.py --config $config
# else
#     python3 mejiro/pipeline/_06_h5_export.py --config $config
# fi
# echo 'h5 file generation complete.'