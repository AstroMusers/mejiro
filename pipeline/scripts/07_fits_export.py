import getpass
import hydra
import numpy as np
import os
import platform
import sys
import time
from astropy.io import fits
from datetime import datetime
from glob import glob
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir
    data_dir = config.machine.data_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util
    from mejiro.lenses import lens_util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    # set nice level
    os.nice(pipeline_params['nice'])

    # set output path
    output_dir = os.path.join(data_dir, 'fits_export')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # get parameters
    survey_params = util.hydra_to_dict(config.survey)
    snr_band = survey_params['snr_band']
    bands = pipeline_params['bands']

    # get all lenses
    all_lenses = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=True, verbose=True, limit=None,
                                                 exposure=True)

    for lens in tqdm(all_lenses):
        uid = lens.uid

        # set output filepath
        fits_path = os.path.join(output_dir, f'roman_hlwas_strong_lens_{str(uid).zfill(8)}.fits')

        # build primary header
        primary_header = fits.Header()

        # general info
        primary_header['VERSION'] = (mejiro.__version__, 'mejiro version')
        primary_header['AUTHOR'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
        primary_header['CREATED'] = (datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
        primary_header['UID'] = (lens.uid, 'UID for system assigned by mejiro')

        # lens params
        primary_header['ZSOURCE'] = (lens.z_source, 'Source galaxy redshift')
        primary_header['ZLENS'] = (lens.z_lens, 'Lens galaxy redshift')
        primary_header['DS'] = (lens.d_s, 'Comoving distance to source galaxy [Gpc]')
        primary_header['DL'] = (lens.d_l, 'Comoving distance to lens galaxy [Gpc]')
        primary_header['DLS'] = (lens.d_ls, 'Comoving distance between lens and source [Gpc]')
        primary_header['HALOMASS'] = (lens.main_halo_mass, 'Lens galaxy main halo mass [M_sun]')
        primary_header['THETAE'] = (lens.get_einstein_radius(), 'Einstein radius [arcsec]')
        primary_header['SIGMAV'] = (lens.lens_vel_disp, 'Lens galaxy velocity dispersion [km/s]')
        primary_header['MU'] = (lens.magnification, 'Flux-weighted magnification of source')
        primary_header['NUMIMAGE'] = (lens.num_images, 'Number of images formed by the system')
        primary_header['SNR'] = (lens.snr, f'Signal-to-noise ratio in {snr_band} band')

        # subhalo params
        primary_header['LOGMLOW'] = (pipeline_params['log_mlow'], 'Lower mass limit for subhalos [log10(M_sun)]')
        primary_header['LOGMHIGH'] = (pipeline_params['log_mhigh'], 'Upper mass limit for subhalos [log10(M_sun)]')
        primary_header['RTIDAL'] = (pipeline_params['r_tidal'], 'See pyHalo documentation')
        primary_header['SIGMASUB'] = (pipeline_params['sigma_sub'], 'See pyHalo documentation')
        primary_header['NUMSUB'] = (lens.num_subhalos, 'Number of subhalos')

        primary_hdu = fits.PrimaryHDU(None, primary_header)
        hdul = fits.HDUList([primary_hdu])

        # get images
        image_paths = sorted(glob(f'{pipeline_dir}/04/**/galsim_{str(uid).zfill(8)}_*.npy'))
        assert len(image_paths) == len(bands), 'Could not find an image for each band'
        images = [np.load(image_path) for image_path in image_paths]

        for band, image in zip(bands, images):
            header = fits.Header()

            header['INSTRUME'] = ('WFI', 'Instrument')
            header['FILTER'] = (band, 'Filter')
            header['EXPOSURE'] = (pipeline_params['exposure_time'], 'Exposure time [seconds]')
            header['OVERSAMP'] = (pipeline_params['grid_oversample'], 'Oversampling used in calculation')
            header['PIXELSCL'] = (0.11, 'Pixel scale [arcsec/pixel]')
            header['FOV'] = (0.11 * pipeline_params['final_pixel_side'], 'Field of view [arcsec]')
            header['DETECTOR'] = (lens.detector, 'Detector')
            header['DET_X'] = (lens.detector_position[0], 'Detector X position')
            header['DET_Y'] = (lens.detector_position[1], 'Detector Y position')

            # lens params
            header['SOURCMAG'] = (lens.source_mags[band], 'Unlensed source galaxy AB magnitude')
            header['MAGNMAG'] = (lens.lensed_source_mags[band], 'Lensed source galaxy AB magnitude')
            header['LENSMAG'] = (lens.lens_mags[band], 'Lens galaxy AB magnitude')

            image_hdu = fits.ImageHDU(image, header, name=band)
            hdul.append(image_hdu)

        hdul.writeto(fits_path, overwrite=True)

    stop = time.time()
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    main()
