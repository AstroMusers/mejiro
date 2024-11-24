import os
import sys
import time
from datetime import datetime
from glob import glob

import getpass
import h5py
import platform
import hydra
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import galsim
import lenstronomy
import webbpsf


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir
    data_dir = config.machine.data_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import roman_util, util
    from mejiro.lenses import lens_util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = False  # retrieve from prod
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    # set nice level
    os.nice(pipeline_params['nice'])

    # get parameters
    survey_params = util.hydra_to_dict(config.survey)
    snr_band = survey_params['snr_band']
    bands = pipeline_params['bands']

    # create h5 file
    filepath = f'{data_dir}/h5_export/roman_hlwas_v_0_0_2.h5'
    if os.path.exists(filepath):
        os.remove(filepath)
    f = h5py.File(filepath, 'a')  # append mode: read/write if exists, create otherwise

    # set file-level attributes
    f.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    f.attrs['author'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    f.attrs['created'] = (now_string)

    # ---------------------------CREATE IMAGE DATASET--------------------------------
    group_images = f.create_group('images')

    # set group-level attributes
    group_images.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    group_images.attrs['lenstronomy_version'] = (lenstronomy.__version__, 'lenstronomy version')
    group_images.attrs['galsim_version'] = (galsim.__version__, 'GalSim version')
    group_images.attrs['webbpsf_version'] = (webbpsf.__version__, 'WebbPSF version')

    # get all lenses
    all_lenses = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=True, verbose=True, limit=None,
                                                 exposure=True)

    print(f'Creating datasets for {len(all_lenses)} lenses')
    for lens in tqdm(all_lenses):
        uid = lens.uid

        # create group for StrongLens
        group_lens = group_images.create_group(f'strong_lens_{str(uid).zfill(8)}')

        # set group-level attributes
        group_lens.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
        group_lens.attrs['uid'] = (uid, 'Unique identifier for system assigned by mejiro')
        group_lens.attrs['z_source'] = (str(lens.z_source), 'Source galaxy redshift')
        group_lens.attrs['z_lens'] = (str(lens.z_lens), 'Lens galaxy redshift')
        group_lens.attrs['d_s'] = (str(lens.d_s), 'Comoving distance to source galaxy [Gpc]')
        group_lens.attrs['d_l'] = (str(lens.d_l), 'Comoving distance to lens galaxy [Gpc]')
        group_lens.attrs['d_ls'] = (str(lens.d_ls), 'Comoving distance between lens and source [Gpc]')
        group_lens.attrs['main_halo_mass'] = (str(lens.main_halo_mass), 'Lens galaxy main halo mass [M_sun]')
        group_lens.attrs['theta_e'] = (str(lens.get_einstein_radius()), 'Einstein radius [arcsec]')
        group_lens.attrs['sigma_v'] = (str(lens.lens_vel_disp), 'Lens galaxy velocity dispersion [km/s]')
        group_lens.attrs['mu'] = (str(lens.magnification), 'Flux-weighted magnification of source')
        group_lens.attrs['num_images'] = (str(lens.num_images), 'Number of images formed by the system')
        group_lens.attrs['snr'] = (str(lens.snr), f'Signal-to-noise ratio in {snr_band} band')
        group_lens.attrs['instrument'] = ('WFI', 'Instrument')
        group_lens.attrs['exposure_time'] = (str(pipeline_params['exposure_time']), 'Exposure time [seconds]')
        # group_lens.attrs['supersampling_factor'] = (pipeline_params['grid_oversample'], 'Supersampling factor used in calculation')
        group_lens.attrs['detector'] = (str(lens.detector), 'Detector')
        group_lens.attrs['detector_position_x'] = (str(lens.detector_position[0]), 'Detector X position')
        group_lens.attrs['detector_position_y'] = (str(lens.detector_position[1]), 'Detector Y position')
        group_lens.attrs['log_mlow'] = (
        str(pipeline_params['log_mlow']), 'Lower mass limit for subhalos [log10(M_sun)]')
        group_lens.attrs['log_mhigh'] = (
        str(pipeline_params['log_mhigh']), 'Upper mass limit for subhalos [log10(M_sun)]')
        group_lens.attrs['r_tidal'] = (str(pipeline_params['r_tidal']), 'See pyHalo documentation')
        group_lens.attrs['sigma_sub'] = (str(pipeline_params['sigma_sub']), 'See pyHalo documentation')
        group_lens.attrs['num_subhalos'] = (str(lens.num_subhalos), 'Number of subhalos')

        # create dataset for SNR array
        dataset_snr_array = group_lens.create_dataset(f'snr_array_{str(uid).zfill(8)}', data=lens.masked_snr_array)

        for i, band in enumerate(pipeline_params['bands']):
            # load array and image
            # array_path = f'{pipeline_dir}/03/**/array_{str(uid).zfill(8)}_{band}.npy'
            image_path = f'{pipeline_dir}/04/**/galsim_{str(uid).zfill(8)}_{band}.npy'
            # array = np.load(glob(array_path)[0])
            image = np.load(glob(image_path)[0])

            # create datasets
            # dataset_synth = group_lens.create_dataset(f'synthetic_image_{str(uid).zfill(8)}_{band}', data=array)
            dataset_exposure = group_lens.create_dataset(f'exposure_{str(uid).zfill(8)}_{band}', data=image)

            # set synthetic image dataset attributes
            # dataset_synth.attrs['pixel_scale'] = (str(0.11 / pipeline_params['grid_oversample']), 'Pixel scale [arcsec/pixel]')
            # dataset_synth.attrs['fov'] = (str(pipeline_params['side']), 'Field of view [arcsec]')

            # set exposure dataset attributes
            dataset_exposure.attrs['pixel_scale'] = (str(0.11), 'Pixel scale [arcsec/pixel]')
            dataset_exposure.attrs['fov'] = (str(0.11 * pipeline_params['final_pixel_side']), 'Field of view [arcsec]')

            # attributes to set on both
            for dset in [dataset_exposure]:  # dataset_synth,
                dset.attrs['units'] = ('Counts/sec', 'Units of pixel values')
                dset.attrs['filter'] = (band, 'Filter')
                dset.attrs['source_magnitude'] = (str(lens.source_mags[band]), 'Unlensed source galaxy AB magnitude')
                dset.attrs['lensed_source_magnitude'] = (
                str(lens.lensed_source_mags[band]), 'Lensed source galaxy AB magnitude')
                dset.attrs['lens_magnitude'] = (str(lens.lens_mags[band]), 'Lens galaxy AB magnitude')

    # ---------------------------CREATE PSF DATASET--------------------------------
    # set detectors and detector_positions
    detectors = range(1, 19)
    detector_positions = roman_util.divide_up_sca(5)

    # create psfs dataset
    group_psfs = f.create_group('psfs')

    # set group-level attributes
    group_psfs.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
    group_psfs.attrs['webbpsf_version'] = (webbpsf.__version__, 'WebbPSF version')

    for det in detectors:
        print(f'Adding PSFs for detector {det}')

        # create group for detector
        group_detector = group_psfs.create_group(f'sca{str(det).zfill(2)}')

        for det_pos in tqdm(detector_positions):
            for i, band in enumerate(pipeline_params['bands']):
                psf_image = util.unpickle(
                    f'/data/bwedig/mejiro/cached_psfs/{band}_{det}_{det_pos[0]}_{det_pos[1]}_5_101.pkl')
                psf = psf_image.image.array

                dataset_psf = group_detector.create_dataset(f'psf_{det}_{det_pos[0]}_{det_pos[1]}_{band}', data=psf)

                # set psf dataset attributes
                dataset_psf.attrs['mejiro_version'] = (mejiro.__version__, 'mejiro version')
                dataset_psf.attrs['detector'] = (str(det), 'Detector')
                dataset_psf.attrs['detector_position_x'] = (str(det_pos[0]), 'Detector X position')
                dataset_psf.attrs['detector_position_y'] = (str(det_pos[1]), 'Detector Y position')
                dataset_psf.attrs['fov_pixels'] = (str(101), 'See WebbPSF documentation')
                dataset_psf.attrs['oversample'] = (str(pipeline_params['grid_oversample']), 'See WebbPSF documentation')

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
