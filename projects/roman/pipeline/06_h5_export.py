import argparse
import galsim
import getpass
import h5py
import lenstronomy
import numpy as np
import os
import platform
import time
import yaml
import stpsf
import slsim
from datetime import datetime
from glob import glob
from tqdm import tqdm

import mejiro
from mejiro.utils import roman_util, util


PREV_SCRIPT_NAME = '05'
SCRIPT_NAME = '06'


def main(args):
    start = time.time()

    # ensure the configuration file has a .yaml or .yml extension
    if not args.config.endswith(('.yaml', '.yml')):
        if os.path.exists(args.config + '.yaml'):
            args.config += '.yaml'
        elif os.path.exists(args.config + '.yml'):
            args.config += '.yml'
        else:
            raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")

    # read configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    limit = config['limit']
    bands = config['synthetic_image']['bands']
    imaging_config = config['imaging']
    subhalo_config = config['subhalos']
    snr_config = config['snr']
    psf_config = config['psf']
    dataset_config = config['output']

    # set up top directory for all pipeline output
    pipeline_dir = os.path.join(data_dir, config['pipeline_dir'])
    if dev:
        pipeline_dir += '_dev'

    # tell script where the output of previous script is
    input_dir = os.path.join(pipeline_dir, '05')

    # parse output of previous script to determine which SCAs to process
    sca_dir_names = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'sca*')) if os.path.isdir(d)]
    scas = sorted([int(d[3:]) for d in sca_dir_names])
    scas = [str(sca).zfill(2) for sca in scas]

    # get lens UIDs, organized by SCA
    uid_dict = {}
    for sca in scas:
        pickled_exposures = sorted(glob(input_dir + f'/sca{sca}/Exposure_*.pkl'))
        lens_uids = [os.path.basename(i).split('_')[1] for i in pickled_exposures]
        lens_uids = list(set(lens_uids))  # remove duplicates, e.g., if multiple bands
        lens_uids = sorted(lens_uids)
        uid_dict[sca] = lens_uids  # for each SCA, list of UIDs of associated lenses

    if verbose: print(f'Found {len(uid_dict)} SCA(s) with {sum([len(v) for v in uid_dict.values()])} lenses')

    # create h5 file
    dataset_version = str(dataset_config['version'])
    version_string = dataset_version.replace('.', '_')
    filepath = os.path.join(data_dir, f'{pipeline_dir}_v_{version_string}.h5')
    if os.path.exists(filepath):
        os.remove(filepath)
    f = h5py.File(filepath, 'a')  # append mode: read/write if exists, create otherwise

    # set file-level attributes
    f.attrs['author'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    f.attrs['created'] = (now_string)
    f.attrs['dataset_version'] = (dataset_version)
    f.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')

    # ---------------------------CREATE IMAGE DATASET--------------------------------
    group_images = f.create_group('images')

    # set group-level attributes
    group_images.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    group_images.attrs['lenstronomy_version'] = (lenstronomy.__version__, 'lenstronomy version')
    group_images.attrs['slsim_version'] = (slsim.__version__, 'SLSim version')
    group_images.attrs['stpsf_version'] = (stpsf.__version__, 'STPSF version')
    group_images.attrs['galsim_version'] = (galsim.__version__, 'GalSim version')

    for sca, uid_list in tqdm(uid_dict.items(), desc=f'SCAs', position=0, leave=True):
        for uid in tqdm(uid_list, desc='Strong Lenses', position=1, leave=False):

            # create group for StrongLens
            group_lens = group_images.create_group(f'strong_lens_{str(uid).zfill(8)}')

            exposure_pickles = sorted(glob(input_dir + f'/sca{sca}/Exposure_{uid}_*.pkl'))

            # grab an exposure and strong lens
            exposure = util.unpickle(exposure_pickles[0])
            synthetic_image = exposure.synthetic_image
            lens = synthetic_image.strong_lens

            # set group-level attributes
            # group_lens.attrs['uid'] = (uid, 'Unique identifier for system assigned by mejiro')
            group_lens.attrs['z_source'] = (str(lens.z_source), 'Source galaxy redshift')
            group_lens.attrs['z_lens'] = (str(lens.z_lens), 'Lens galaxy redshift')
            group_lens.attrs['main_halo_mass'] = (str(lens.get_main_halo_mass()), 'Lens galaxy main halo mass [M_sun]')
            group_lens.attrs['theta_e'] = (str(lens.get_einstein_radius()), 'Einstein radius [arcsec]')
            # group_lens.attrs['sigma_v'] = (str(lens.lens_vel_disp), 'Lens galaxy velocity dispersion [km/s]')
            # group_lens.attrs['mu'] = (str(lens.magnification), 'Flux-weighted magnification of source')
            # group_lens.attrs['num_images'] = (str(lens.num_images), 'Number of images formed by the system')
            # group_lens.attrs['snr'] = (str(lens.snr), f'Signal-to-noise ratio in {snr_band} band')
            # group_lens.attrs['instrument'] = ('WFI', 'Instrument')
            # group_lens.attrs['exposure_time'] = (str(pipeline_params['exposure_time']), 'Exposure time [seconds]')
            # group_lens.attrs['supersampling_factor'] = (pipeline_params['grid_oversample'], 'Supersampling factor used in calculation')
            group_lens.attrs['detector'] = (str(synthetic_image.instrument_params['detector']), 'Detector')
            group_lens.attrs['detector_position_x'] = (str(synthetic_image.instrument_params['detector_position'][0]), 'Detector X position')
            group_lens.attrs['detector_position_y'] = (str(synthetic_image.instrument_params['detector_position'][1]), 'Detector Y position')

            # set attributes for subhalos
            if lens.realization is not None:
                group_lens.attrs['subhalos'] = ('True', 'Are subhalos present in this lens?')
                group_lens.attrs['log_mlow'] = (
                    str(subhalo_config['log_mlow']), 'Lower mass limit for subhalos [log10(M_sun)]')
                group_lens.attrs['log_mhigh'] = (
                    str(subhalo_config['log_mhigh']), 'Upper mass limit for subhalos [log10(M_sun)]')
                group_lens.attrs['r_tidal'] = (str(subhalo_config['r_tidal']), 'See pyHalo documentation')
                group_lens.attrs['sigma_sub'] = (str(subhalo_config['sigma_sub']), 'See pyHalo documentation')
                group_lens.attrs['num_subhalos'] = (str(len(lens.realization.halos)), 'Number of subhalos')
            else:
                group_lens.attrs['subhalos'] = ('False', 'Are subhalos present in this lens?')

            # create dataset for SNR array
            # dataset_snr_array = group_lens.create_dataset(f'snr_array_{str(uid).zfill(8)}', data=lens.masked_snr_array)

            for i, band in enumerate(bands):
                # load exposure
                exposure = util.unpickle(exposure_pickles[i])
                synthetic_image = exposure.synthetic_image

                # create datasets
                dataset_synth = group_lens.create_dataset(f'synthetic_image_{str(uid).zfill(8)}_{band}', data=synthetic_image.image)
                dataset_exposure = group_lens.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}', data=exposure.exposure)

                # set synthetic image dataset attributes
                dataset_synth.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
                dataset_synth.attrs['fov'] = (str(synthetic_image.fov_arcsec), 'Field of view [arcsec]')

                # set exposure dataset attributes
                dataset_exposure.attrs['pixel_scale'] = (str(synthetic_image.pixel_scale), 'Pixel scale [arcsec/pixel]')
                dataset_exposure.attrs['fov'] = (str(round(synthetic_image.pixel_scale * exposure.exposure.shape[0], 2)), 'Field of view [arcsec]')

                # attributes to set on both
                for dset in [dataset_exposure]:  # dataset_synth,
                    dset.attrs['units'] = ('Counts/sec', 'Units of pixel values')
                    dset.attrs['filter'] = (band, 'Filter')
                    dset.attrs['source_magnitude'] = (str(lens.physical_params['magnitudes']['source'][band]), 'Unlensed source galaxy magnitude')
                    dset.attrs['lensed_source_magnitude'] = (
                        str(lens.physical_params['magnitudes']['lensed_source'][band]), 'Lensed source galaxy magnitude')
                    dset.attrs['lens_magnitude'] = (str(lens.physical_params['magnitudes']['lens'][band]), 'Lens galaxy magnitude')

    # ---------------------------CREATE PSF DATASET--------------------------------
    # set detectors and detector_positions
    detectors = psf_config['detectors']
    detector_positions = roman_util.divide_up_sca(psf_config['divide_up_sca'])

    # hard-coded PSF params, for now
    psf_pixels = 101
    psf_oversample = 5

    # create psfs dataset
    group_psfs = f.create_group('psfs')

    # set group-level attributes
    group_psfs.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    group_psfs.attrs['stpsf_version'] = (stpsf.__version__, 'STPSF version')

    # retrieve configuration settings
    psf_cache_dir = os.path.join(data_dir, config['psf_cache_dir'])

    for det in tqdm(detectors, desc='Detectors', position=0, leave=True):
        # create group for detector
        group_detector = group_psfs.create_group(f'sca{str(det).zfill(2)}')

        for det_pos in tqdm(detector_positions, desc='Detector Positions', position=1, leave=False):
            for i, band in enumerate(bands):
                psf = np.load(os.path.join(psf_cache_dir, f'{band}_{det}_{det_pos[0]}_{det_pos[1]}_{psf_oversample}_{psf_pixels}.npy'))

                dataset_psf = group_detector.create_dataset(f'psf_{det}_{det_pos[0]}_{det_pos[1]}_{band}', data=psf)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
