#!/bin/bash

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

# escape if error encountered
set -e

# execute Python scripts sequentially
echo 'HLWAS COADD START'
python3 pipeline/scripts/00_sim_hlwas_coadd.py
echo 'HLWAS COADD END'

echo 'HLTDS DEEP START'
python3 pipeline/scripts/00_sim_hltds_deep.py
echo 'HLTDS DEEP END'

echo 'HLTDS WIDE START'
python3 pipeline/scripts/00_sim_hltds_wide.py
echo 'HLTDS WIDE END'