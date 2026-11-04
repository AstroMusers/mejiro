#!/bin/bash

# this bash script is a less-polished way of executing the `mejiro` pipeline that is useful for development

# change directory to the root of the repository
cd "$(dirname "$0")/.."

# usage: examples/execute_pipeline.sh [config] [data_dir]
#   config:   config file, relative to the repository root (default: test.yaml)
#   data_dir: output directory, overriding the config's data_dir (needed when the config
#             leaves data_dir unset, as test.yaml does)
config="${1:-mejiro/data/mejiro_config/test.yaml}"
data_dir_args=()
if [ -n "${2:-}" ]; then
    data_dir_args=(--data_dir "$2")
fi

# escape if error encountered
set -e

# execute Python scripts sequentially
echo 'Caching PSFs...'
python3 mejiro/pipeline/_00_cache_psfs.py --config "$config" "${data_dir_args[@]}"
echo 'Cached PSFs.'

echo 'Generating galaxy tables...'
python3 mejiro/pipeline/_01a_generate_galaxy_tables.py --config "$config" "${data_dir_args[@]}"
echo 'Generated galaxy tables.'

echo 'Running survey simulation...'
python3 mejiro/pipeline/_01b_run_survey_simulation.py --config "$config" "${data_dir_args[@]}"
echo 'Identified detectable strong lenses.'

echo 'Building lens list from SkyPy...'
python3 mejiro/pipeline/_02_build_lens_list.py --config "$config" "${data_dir_args[@]}"
echo 'Built lens list.'

echo 'Adding subhalos with PyHalo...'
python3 mejiro/pipeline/_03_generate_subhalos.py --config "$config" "${data_dir_args[@]}"
echo 'Added subhalos.'

echo 'Building models...'
python3 mejiro/pipeline/_04_create_synthetic_images.py --config "$config" "${data_dir_args[@]}"
echo 'Built models.'

# echo 'Simulating images...'
# python3 mejiro/pipeline/_05_galsim.py --config "$config" "${data_dir_args[@]}"
# echo 'GalSim simulations complete.'

echo 'Simulating images...'
python3 mejiro/pipeline/_05_romanisim.py --config "$config" "${data_dir_args[@]}"
echo 'romanisim simulations complete.'

echo 'Calculating SNRs...'
python3 mejiro/pipeline/calculate_snrs.py --config "$config" "${data_dir_args[@]}"
echo 'SNR calculation complete.'

echo 'Generating h5 file...'
python3 mejiro/pipeline/_06_h5_export.py --config "$config" "${data_dir_args[@]}"
echo 'h5 file generation complete.'