"""
Exports romanisim exposures and synthetic images to HDF5 format.

This script reads romanisim exposure cutouts (.npy) and the corresponding SyntheticImage
pickles from previous pipeline steps, and writes them to an HDF5 file with relevant
metadata. Since romanisim output does not produce Exposure objects, certain fields are
sourced differently:

    - exposure data: loaded from .npy cutout files (05_romanisim/sca*/)
    - exposure_time: from config['imaging']['exposure_time']
    - pixel_scale: from the SyntheticImage object (04/sca*/)
    - SNR: not available (romanisim does not provide separate source/lens channels)
    - units: DN/s

All lens and synthetic image metadata (redshifts, Einstein radius, magnitudes, detector
info, etc.) is read from the original SyntheticImage pickles in step 04.

Usage:
    python3 _06_h5_export_romanisim.py --config <config.yaml>

Arguments:
    --config: Path to the YAML configuration file.
"""
import argparse
import getpass
import h5py
import lenstronomy
import os
import pandas as pd
import platform
import romanisim
import slsim
import stpsf
import time
from datetime import datetime
from glob import glob
from tqdm import tqdm

import logging

import numpy as np

from romanisim import parameters as romanisim_params

import mejiro
from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)

PREV_SCRIPT_NAME = '05_romanisim'
SCRIPT_NAME = '06'
SUPPORTED_INSTRUMENTS = ['roman']


def main(args):
    start = time.time()

    # initialize PipelineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS)

    # determine if labeled dataset
    labeled = False

    # retrieve configuration parameters
    bands = pipeline.config['synthetic_image']['bands']
    subhalo_config = pipeline.config['subhalos']
    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']
    dataset_config = pipeline.config['dataset']

    # compute exposure time from romanisim MA table
    ma_table_number = pipeline.config['exposure']['ma_table_number']
    read_pattern = romanisim_params.read_pattern[ma_table_number]
    exposure_time = romanisim_params.read_time * read_pattern[-1][-1]

    # discover exposure cutouts and parse UIDs
    exposure_npy_files = sorted(glob(os.path.join(pipeline.input_dir, 'sca*', f'Exposure_{pipeline.name}_*.npy')))
    uids = set()
    for f in exposure_npy_files:
        basename = os.path.basename(f)
        uid = basename.split("_")[-2]
        uids.add(uid)
    uids = sorted(uids)
    logger.info(f'Found {len(uids)} unique systems')

    # calculate and implement limit, if specified
    if pipeline.limit is not None:
        uids = uids[:pipeline.limit]
        logger.info(f'Limiting to {len(uids)} systems')

    # directory containing SyntheticImage pickles from step 04
    synth_input_dir = os.path.join(pipeline.pipeline_dir, '04')

    # create h5 file
    dataset_version = str(dataset_config['version'])
    version_string = dataset_version.replace('.', '_')
    filepath = os.path.join(pipeline.output_dir, f'{pipeline.name}_v_{version_string}.h5')
    if os.path.exists(filepath):
        os.remove(filepath)
    f = h5py.File(filepath, 'a')

    # if not labeled, export the "answer key" file
    if not labeled:
        answer_key_filepath = os.path.join(pipeline.output_dir, f'{pipeline.name}_answer_key_v_{version_string}.csv')
        if os.path.exists(answer_key_filepath):
            os.remove(answer_key_filepath)

        df = pd.DataFrame(columns=['uid', 'einstein_radius'])

    # set file-level attributes
    f.attrs['author'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    f.attrs['created'] = (now_string)
    f.attrs['dataset_version'] = (dataset_version)
    f.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    f.attrs['lenstronomy_version'] = (lenstronomy.__version__, 'lenstronomy version')
    f.attrs['slsim_version'] = (slsim.__version__, 'SLSim version')
    f.attrs['romanisim_version'] = (romanisim.__version__, 'romanisim version')
    f.attrs['stpsf_version'] = (stpsf.__version__, 'STPSF version')

    # ---------------------------CREATE IMAGE DATASET--------------------------------
    group_images = f.create_group('images')

    for uid in tqdm(uids):
        group_lens = group_images.create_group(f'strong_lens_{str(uid).zfill(8)}')

        # load SyntheticImage pickles to get metadata
        synth_pickles = sorted(glob(os.path.join(synth_input_dir, f'sca*/SyntheticImage_{pipeline.name}_{uid}_*.pkl')))
        if not synth_pickles:
            logger.warning(f'No SyntheticImage pickles found for UID {uid}, skipping')
            continue

        synthetic_image = util.unpickle(synth_pickles[0])
        lens = synthetic_image.strong_lens

        if not labeled:
            df.loc[len(df)] = [uid, lens.get_einstein_radius()]

        # set group-level attributes
        group_lens.attrs['uid'] = (uid, 'Unique identifier for system assigned by mejiro')
        group_lens.attrs['z_source'] = (str(lens.z_source), 'Source galaxy redshift')
        group_lens.attrs['z_lens'] = (str(lens.z_lens), 'Lens galaxy redshift')
        if labeled:
            group_lens.attrs['main_halo_mass'] = (str(lens.get_main_halo_mass()), 'Lens galaxy main halo mass [M_sun]')
            group_lens.attrs['theta_e'] = (str(lens.get_einstein_radius()), 'Einstein radius [arcsec]')
            group_lens.attrs['sigma_v'] = (str(lens.get_velocity_dispersion()), 'Lens galaxy velocity dispersion [km/s]')
            group_lens.attrs['mu'] = (str(lens.get_magnification()), 'Flux-weighted magnification of source')
        group_lens.attrs['detector'] = (str(synthetic_image.instrument_params['detector']), 'Detector')
        group_lens.attrs['detector_position_x'] = (str(synthetic_image.instrument_params['detector_position'][0]), 'Detector X position')
        group_lens.attrs['detector_position_y'] = (str(synthetic_image.instrument_params['detector_position'][1]), 'Detector Y position')

        # set attributes for subhalos
        if lens.realization is not None:
            group_lens.attrs['substructure'] = ('True', 'Is substructure present in this lens?')
            for key, value in subhalo_config['realization_kwargs'].items():
                group_lens.attrs[key] = (str(value), 'See pyHalo documentation')
        else:
            group_lens.attrs['substructure'] = ('False', 'Is substructure present in this lens?')

        for i, band in enumerate(bands):
            # load the .npy cutout
            sca_string = f'sca{str(synthetic_image.instrument_params["detector"]).zfill(2)}'
            exposure_npy = os.path.join(pipeline.input_dir, sca_string,
                                        f'Exposure_{pipeline.name}_{uid}_{band}.npy')
            if not os.path.exists(exposure_npy):
                logger.warning(f'Exposure cutout not found: {exposure_npy}, skipping')
                continue
            exposure_data = np.load(exposure_npy)

            # load corresponding SyntheticImage for this band
            if i < len(synth_pickles):
                synthetic_image = util.unpickle(synth_pickles[i])

            # create datasets
            dataset_exposure = group_lens.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}', data=exposure_data)
            dset_list = [dataset_exposure]

            # set exposure dataset attributes
            dataset_exposure.attrs['exposure_time'] = (str(exposure_time), 'Exposure time [seconds]')
            dataset_exposure.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
            dataset_exposure.attrs['fov'] = (str(round(synthetic_image.pixel_scale * exposure_data.shape[0], 2)), 'Field of view [arcsec]')

            if dataset_config['include_synthetic_images']:
                dataset_synth = group_lens.create_dataset(f'synthetic_image_{str(uid).zfill(8)}_{band}', data=synthetic_image.data)
                dset_list.append(dataset_synth)

                dataset_synth.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
                dataset_synth.attrs['fov'] = (str(synthetic_image.fov_arcsec), 'Field of view [arcsec]')

            # attributes to set on both
            for dset in dset_list:
                dset.attrs['units'] = ('DN/s', 'Units of pixel values')
                dset.attrs['filter'] = (band, 'Filter')
                if labeled:
                    dset.attrs['source_magnitude'] = (str(lens.get_source_magnitude(band)), 'Unlensed source galaxy magnitude')
                    dset.attrs['lensed_source_magnitude'] = (
                        str(lens.get_lensed_source_magnitude(band)), 'Lensed source galaxy magnitude')
                    dset.attrs['lens_magnitude'] = (str(lens.get_lens_magnitude(band)), 'Lens galaxy magnitude')

    if dataset_config['include_psfs']:
        # ---------------------------CREATE PSF DATASET--------------------------------
        detectors = psf_config['detectors']
        detector_positions = roman_util.divide_up_sca(psf_config['divide_up_detector'])

        # hard-coded PSF params, for now
        psf_pixels = psf_config['num_pixes'][0]
        psf_oversample = synthetic_image_config['supersampling_factor']

        group_psfs = f.create_group('psfs')

        for det in tqdm(detectors, desc='Detectors', position=0, leave=True):
            group_detector = group_psfs.create_group(f'sca{str(det).zfill(2)}')

            for det_pos in tqdm(detector_positions, desc='Detector Positions', position=1, leave=False):
                for i, band in enumerate(bands):
                    psf_id_string = STPSFEngine.get_psf_id(band, det, det_pos, psf_oversample, psf_pixels)
                    psf = STPSFEngine.get_cached_psf(psf_id_string, pipeline.psf_cache_dir)

                    if psf is None:
                        logger.warning(f'Cached PSF not found for {psf_id_string}. Skipping PSF dataset creation.')
                        continue

                    dataset_psf = group_detector.create_dataset(f'psf_{psf_id_string}', data=psf)

                    dataset_psf.attrs['detector'] = (str(det), 'Detector')
                    dataset_psf.attrs['detector_position_x'] = (str(det_pos[0]), 'Detector X position')
                    dataset_psf.attrs['detector_position_y'] = (str(det_pos[1]), 'Detector Y position')
                    dataset_psf.attrs['fov_pixels'] = (str(psf_pixels), 'See STPSF documentation')
                    dataset_psf.attrs['oversample'] = (str(psf_oversample), 'See STPSF documentation')

    # if not labeled, save answer key
    if not labeled:
        df.to_csv(answer_key_filepath, index=False)
        logger.info(f'Wrote answer key to {answer_key_filepath}')

    stop = time.time()
    util.print_execution_time(start, stop)

    logger.info(f'Wrote dataset to {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export romanisim dataset to HDF5 format.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
