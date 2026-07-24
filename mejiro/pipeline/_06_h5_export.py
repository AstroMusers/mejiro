"""
Exports step-05 exposures and synthetic images to HDF5 format.

This script reads the exposure cutouts written by whichever step-05 variant ran, plus the
corresponding SyntheticImage files from step 04, and writes them to an HDF5 file with
relevant metadata. The input step is given by --prev-step, and it determines both the file
format and the pixel units:

    - exposure data: read via util.load_exposure -- bare arrays from 05_romanisim/sca*/Exposure_*.npy,
      or the .data of the lightweight .npz / full .pkl Exposure written to 05_galsim/sca*/
    - exposure_time: for romanisim, from config['exposure']['ma_table_number'] via romanisim
      parameters; for galsim, config['imaging']['exposure_time']
    - units: galsim writes DN (counts; Roman gain is 1.0 e-/DN). romanisim writes L2 in DN/s
      and L3 in MJy/sr; since both levels land in the same directory, the level is read from
      the exposure_level.txt sidecar that _05_romanisim writes.
    - lens and synthetic image metadata: from the SyntheticImage files in step 04
    - SNR: read from name_snr_pairs.pkl produced by calculate_snrs.py (optional)

If config['dataset']['labeled'] is False, truth attributes (main halo mass, Einstein
radius, velocity dispersion, substructure) are omitted from the HDF5 file and an
answer-key CSV is written alongside it.

Usage:
    python3 _06_h5_export.py --config <config.yaml> [--data_dir <dir>] [--prev-step <step_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Parent directory of pipeline output. Overrides data_dir in the config file.
    --prev-step: Pipeline step directory holding the input exposures (default: '05_romanisim').
"""
import argparse
import getpass
import h5py
import lenstronomy
import numpy as np
import os
import pandas as pd
import platform
import romanisim
import romancal
import slsim
import stpsf
import time
from datetime import datetime
from glob import glob
from tqdm import tqdm

import logging

from romanisim import parameters as romanisim_params

import mejiro
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)

PREV_SCRIPT_NAME = '05_romanisim'
SCRIPT_NAME = '06'
SUPPORTED_INSTRUMENTS = ['roman']


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    # initialize PipelineHelper, resolving which step produced the exposures (default matches
    # the standard romanisim step; override for variants like the sub-pixel L3 run)
    prev_script_name = getattr(args, 'prev_step', None) or PREV_SCRIPT_NAME
    pipeline = PipelineHelper(args, prev_script_name, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # retrieve configuration parameters
    bands = pipeline.config['synthetic_image']['bands']
    subhalo_config = pipeline.config['subhalos']
    dataset_config = pipeline.config['dataset']
    labeled = dataset_config['labeled']

    # exposure time and pixel units both depend on which step-05 variant wrote the input
    extension = PipelineHelper.exposure_extension(prev_script_name, pipeline.config['imaging']['serialization'])
    if extension == '.npy':
        # romanisim: exposure time comes from the MA table, units from the data level.
        # Every level writes to the same directory, so the level is read from the sidecar
        # _05_romanisim leaves behind.
        ma_table_number = pipeline.config['exposure']['ma_table_number']
        read_pattern = romanisim_params.read_pattern[ma_table_number]
        exposure_time = romanisim_params.read_time * read_pattern[-1][-1]
        with open(os.path.join(pipeline.input_dir, 'exposure_level.txt')) as f:
            level = f.read().strip()
        units = 'DN/s' if level == 'l2' else 'MJy/sr'
    else:
        # galsim: counts (= DN for Roman, where gain is 1.0 e-/DN), see mejiro.exposure.Exposure
        exposure_time = pipeline.config['imaging']['exposure_time']
        units = 'DN'

    # discover exposure cutouts and parse UIDs
    logger.info(f'Looking for exposure cutouts in {pipeline.input_dir}')
    exposure_files = sorted(glob(os.path.join(pipeline.input_dir, 'sca*', f'Exposure_{pipeline.name}_*{extension}')))
    uids = sorted({os.path.basename(f).split('_')[-2] for f in exposure_files})
    logger.info(f'Found {len(uids)} unique system(s) ({extension}, {units})')

    # calculate and implement limit, if specified
    if pipeline.limit is not None:
        uids = uids[:pipeline.limit]
        logger.warn(f'Limiting to {len(uids)} systems')

    # directory containing SyntheticImage files from step 04
    synth_input_dir = pipeline.step_dir('04')

    # load SNR lookup from calculate_snrs.py output, if available
    snr_lookup = {}
    snr_pairs_path = os.path.join(pipeline.pipeline_dir, 'snr', 'name_snr_pairs.pkl')
    if os.path.exists(snr_pairs_path):
        snr_lookup = dict(util.unpickle(snr_pairs_path))
        logger.info(f'Reading {len(snr_lookup)} SNRs from {snr_pairs_path}')
    else:
        logger.warning(f'SNR pickle not found at {snr_pairs_path}; SNR attributes will be omitted')

    # create h5 file
    dataset_version = str(dataset_config['version'])
    version_string = dataset_version.replace('.', '_')
    filepath = os.path.join(pipeline.output_dir, f'{pipeline.name}_v_{version_string}.h5')
    f = h5py.File(filepath, 'w')

    # if not labeled, truth values go in an "answer key" CSV instead of the h5
    if not labeled:
        answer_key_filepath = os.path.join(pipeline.output_dir, f'{pipeline.name}_answer_key_v_{version_string}.csv')
        answer_key = pd.DataFrame(columns=['uid', 'einstein_radius', 'substructure_flag', 'deflector_only'])

    # set file-level attributes
    f.attrs['author'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    f.attrs['created'] = (datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    f.attrs['dataset_version'] = (dataset_version)
    f.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    f.attrs['lenstronomy_version'] = (lenstronomy.__version__, 'lenstronomy version')
    f.attrs['slsim_version'] = (slsim.__version__, 'SLSim version')
    f.attrs['romanisim_version'] = (romanisim.__version__, 'romanisim version')
    f.attrs['stpsf_version'] = (stpsf.__version__, 'STPSF version')
    f.attrs['romancal'] = (romancal.__version__, 'romancal version')

    group_images = f.create_group('images')

    for uid in tqdm(uids):
        group_lens = group_images.create_group(f'strong_lens_{str(uid).zfill(8)}')

        # map band -> SyntheticImage file (full .pkl or lightweight .npz)
        synth_files = sorted(
            glob(os.path.join(synth_input_dir, f'sca*/SyntheticImage_{pipeline.name}_{uid}_*.pkl'))
            + glob(os.path.join(synth_input_dir, f'sca*/SyntheticImage_{pipeline.name}_{uid}_*.npz'))
        )
        synth_by_band = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in synth_files}

        synthetic_image = util.load_synthetic_image(synth_files[0])
        lens = synthetic_image.strong_lens

        if not labeled:
            answer_key.loc[len(answer_key)] = [uid, lens.get_einstein_radius(), lens.realization is not None, getattr(synthetic_image, 'deflector_only')]

        # set group-level attributes
        group_lens.attrs['uid'] = (uid, 'Unique identifier for system assigned by mejiro')
        group_lens.attrs['z_source'] = (str(lens.z_source), 'Source galaxy redshift')
        group_lens.attrs['z_lens'] = (str(lens.z_lens), 'Lens galaxy redshift')
        group_lens.attrs['mu'] = (str(lens.get_magnification()), 'Flux-weighted magnification of source')
        group_lens.attrs['detector'] = (str(synthetic_image.instrument_params['detector']), 'Detector')
        group_lens.attrs['detector_position_x'] = (str(synthetic_image.instrument_params['detector_position'][0]), 'Detector X position')
        group_lens.attrs['detector_position_y'] = (str(synthetic_image.instrument_params['detector_position'][1]), 'Detector Y position')

        # set truth attributes, including subhalo params
        if labeled:
            group_lens.attrs['main_halo_mass'] = (str(lens.get_main_halo_mass()), 'Lens galaxy main halo mass [M_sun]')
            group_lens.attrs['theta_e'] = (str(lens.get_einstein_radius()), 'Einstein radius [arcsec]')
            group_lens.attrs['sigma_v'] = (str(lens.get_velocity_dispersion()), 'Lens galaxy velocity dispersion [km/s]')
            if lens.realization is not None:
                group_lens.attrs['substructure'] = ('True', 'Is substructure present in this lens?')
                for key, value in subhalo_config['realization_kwargs'].items():
                    group_lens.attrs[key] = (str(value), 'See pyHalo documentation')
            else:
                group_lens.attrs['substructure'] = ('False', 'Is substructure present in this lens?')
            
            # deflector-only
            group_lens.attrs['deflector_only'] = (
            str(getattr(synthetic_image, 'deflector_only')),
            'Only the deflector (lens galaxy) light was simulated; no source/lensing')

        for band in bands:
            # load the cutout: bare .npy from romanisim, lightweight .npz or full .pkl from
            # galsim. load_exposure dispatches on extension; only .data is needed here.
            sca_string = f'sca{str(synthetic_image.instrument_params["detector"]).zfill(2)}'
            exposure_path = os.path.join(pipeline.input_dir, sca_string,
                                         f'Exposure_{pipeline.name}_{uid}_{band}{extension}')
            exposure_data = util.load_exposure(exposure_path).data

            # load the SyntheticImage for this band
            synthetic_image = util.load_synthetic_image(synth_by_band[band])
            lens = synthetic_image.strong_lens

            # create datasets
            dataset_exposure = group_lens.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}', data=exposure_data)
            dset_list = [dataset_exposure]

            # set exposure dataset attributes
            dataset_exposure.attrs['exposure_time'] = (str(exposure_time), 'Exposure time [seconds]')
            dataset_exposure.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
            dataset_exposure.attrs['fov'] = (str(round(synthetic_image.pixel_scale * exposure_data.shape[0], 2)), 'Field of view [arcsec]')
            snr = snr_lookup.get(f'{pipeline.name}_{uid}_{band}')
            if snr is not None:
                dataset_exposure.attrs['snr'] = (str(snr), 'Signal-to-noise ratio')

            if dataset_config['include_synthetic_images']:
                dataset_synth = group_lens.create_dataset(f'synthetic_image_{str(uid).zfill(8)}_{band}', data=synthetic_image.data)
                dset_list.append(dataset_synth)

                # set synthetic image dataset attributes
                dataset_synth.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
                dataset_synth.attrs['fov'] = (str(synthetic_image.fov_arcsec), 'Field of view [arcsec]')

            # attributes to set on both
            for dset in dset_list:
                dset.attrs['units'] = (units, 'Units of pixel values')
                dset.attrs['filter'] = (band, 'Filter')
                dset.attrs['source_magnitude'] = (str(lens.get_source_magnitude(band)), 'Unlensed source galaxy magnitude')
                dset.attrs['lensed_source_magnitude'] = (
                    str(lens.get_lensed_source_magnitude(band)), 'Lensed source galaxy magnitude')
                dset.attrs['lens_magnitude'] = (str(lens.get_lens_magnitude(band)), 'Lens galaxy magnitude')

    f.close()

    if not labeled:
        answer_key.to_csv(answer_key_filepath, index=False)
        logger.info(f'Wrote answer key to {answer_key_filepath}')

    stop = time.time()
    util.print_execution_time(start, stop)

    logger.info(f'Wrote dataset to {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export the dataset to HDF5 format.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    parser.add_argument('--prev-step', dest='prev_step', type=str, default=None,
                        help=f"Name of the pipeline step directory holding the input exposures "
                             f"(default: '{PREV_SCRIPT_NAME}').")
    args = parser.parse_args()
    main(args)
