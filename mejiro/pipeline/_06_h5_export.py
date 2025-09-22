import argparse
import galsim
import getpass
import h5py
import lenstronomy
import os
import platform
import time
import stpsf
import slsim
from datetime import datetime
from glob import glob
from tqdm import tqdm

import mejiro
from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper


PREV_SCRIPT_NAME = '05'
SCRIPT_NAME = '06'
SUPPORTED_INSTRUMENTS = ['roman']


def main(args):
    start = time.time()

    # initialize PipeLineHelper
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME)

    # retrieve configuration parameters
    bands = pipeline.config['synthetic_image']['bands']
    subhalo_config = pipeline.config['subhalos']
    snr_config = pipeline.config['snr']
    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']
    dataset_config = pipeline.config['dataset']

    # retrieve uids
    if pipeline.instrument_name == 'roman':
        uids = pipeline.parse_roman_uids(prefix='Exposure', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    # create h5 file
    dataset_version = str(dataset_config['version'])
    version_string = dataset_version.replace('.', '_')
    filepath = os.path.join(pipeline.output_dir, f'{pipeline.name}_v_{version_string}.h5')
    if os.path.exists(filepath):
        os.remove(filepath)
    f = h5py.File(filepath, 'a')  # append mode: read/write if exists, create otherwise

    # set file-level attributes
    f.attrs['author'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    f.attrs['created'] = (now_string)
    f.attrs['dataset_version'] = (dataset_version)
    f.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    f.attrs['lenstronomy_version'] = (lenstronomy.__version__, 'lenstronomy version')
    f.attrs['slsim_version'] = (slsim.__version__, 'SLSim version')
    f.attrs['stpsf_version'] = (stpsf.__version__, 'STPSF version')
    f.attrs['galsim_version'] = (galsim.__version__, 'GalSim version')

    # ---------------------------CREATE IMAGE DATASET--------------------------------
    group_images = f.create_group('images')

    for uid in tqdm(uids):
        # create group for StrongLens
        group_lens = group_images.create_group(f'strong_lens_{str(uid).zfill(8)}')

        # unpickle the Exposure
        exposure_pickles = sorted(glob(pipeline.input_dir + f'/sca*/Exposure_{pipeline.name}_{uid}_*.pkl'))  # TODO this is Roman-specific

        # grab an exposure and strong lens
        exposure = util.unpickle(exposure_pickles[0])
        synthetic_image = exposure.synthetic_image
        lens = synthetic_image.strong_lens

        # set group-level attributes
        group_lens.attrs['uid'] = (uid, 'Unique identifier for system assigned by mejiro')
        group_lens.attrs['z_source'] = (str(lens.z_source), 'Source galaxy redshift')
        group_lens.attrs['z_lens'] = (str(lens.z_lens), 'Lens galaxy redshift')
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

        # create dataset for SNR array
        # dataset_snr_array = group_lens.create_dataset(f'snr_array_{str(uid).zfill(8)}', data=lens.masked_snr_array)

        for i, band in enumerate(bands):
            # load exposure
            exposure = util.unpickle(exposure_pickles[i])

            # calculate SNR
            snr = exposure.get_snr(snr_per_pixel_threshold=snr_config['snr_per_pixel_threshold'])

            # create datasets
            dataset_exposure = group_lens.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}', data=exposure.exposure)
            dset_list = [dataset_exposure]

            # set exposure dataset attributes
            dataset_exposure.attrs['snr'] = (str(snr), 'Signal-to-noise ratio')
            dataset_exposure.attrs['exposure_time'] = (str(exposure.exposure_time), 'Exposure time [seconds]')
            dataset_exposure.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
            dataset_exposure.attrs['fov'] = (str(round(synthetic_image.pixel_scale * exposure.exposure.shape[0], 2)), 'Field of view [arcsec]')

            if dataset_config['include_synthetic_images']:
                dataset_synth = group_lens.create_dataset(f'synthetic_image_{str(uid).zfill(8)}_{band}', data=synthetic_image.image)
                dset_list.append(dataset_synth)

                # set synthetic image dataset attributes
                dataset_synth.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
                dataset_synth.attrs['fov'] = (str(synthetic_image.fov_arcsec), 'Field of view [arcsec]')

            # attributes to set on both
            for dset in dset_list:
                dset.attrs['units'] = ('Counts/sec', 'Units of pixel values')
                dset.attrs['filter'] = (band, 'Filter')
                dset.attrs['source_magnitude'] = (str(lens.get_source_magnitude(band)), 'Unlensed source galaxy magnitude')
                dset.attrs['lensed_source_magnitude'] = (
                    str(lens.get_lensed_source_magnitude(band)), 'Lensed source galaxy magnitude')
                dset.attrs['lens_magnitude'] = (str(lens.get_lens_magnitude(band)), 'Lens galaxy magnitude')


    if dataset_config['include_psfs']:
        # ---------------------------CREATE PSF DATASET--------------------------------
        # set detectors and detector_positions
        detectors = psf_config['detectors']
        detector_positions = roman_util.divide_up_sca(psf_config['divide_up_detector'])

        # hard-coded PSF params, for now
        psf_pixels = psf_config['num_pixes'][0]
        psf_oversample = synthetic_image_config['supersampling_factor']

        # create psfs dataset
        group_psfs = f.create_group('psfs')

        # retrieve configuration settings
        psf_cache_dir = os.path.join(pipeline.data_dir, pipeline.config['psf_cache_dir'])

        for det in tqdm(detectors, desc='Detectors', position=0, leave=True):
            # create group for detector
            group_detector = group_psfs.create_group(f'sca{str(det).zfill(2)}')

            for det_pos in tqdm(detector_positions, desc='Detector Positions', position=1, leave=False):
                for i, band in enumerate(bands):
                    # get cached PSF
                    psf_id_string = STPSFEngine.get_psf_id(band, det, det_pos, psf_oversample, psf_pixels)
                    psf = STPSFEngine.get_cached_psf(psf_id_string, psf_cache_dir, verbose=False)

                    # create psf dataset
                    dataset_psf = group_detector.create_dataset(f'psf_{psf_id_string}', data=psf)

                    # set psf dataset attributes
                    dataset_psf.attrs['detector'] = (str(det), 'Detector')
                    dataset_psf.attrs['detector_position_x'] = (str(det_pos[0]), 'Detector X position')
                    dataset_psf.attrs['detector_position_y'] = (str(det_pos[1]), 'Detector Y position')
                    dataset_psf.attrs['fov_pixels'] = (str(psf_pixels), 'See STPSF documentation')
                    dataset_psf.attrs['oversample'] = (str(psf_oversample), 'See STPSF documentation')

    stop = time.time()
    util.print_execution_time(start, stop)

    print(f'Wrote dataset to {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export the dataset to HDF5 format.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
