import os
import sys
import time
from datetime import datetime
from glob import glob

import hydra
import numpy as np
from astropy.io import fits
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    repo_dir = config.machine.repo_dir
    data_dir = config.machine.data_dir

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # set output path
    output_dir = os.path.join(data_dir, 'fits_export')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    now_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    # output_file = os.path.join(config.machine.pipeline_dir, f'mejiro_output_{now_string}.hdf5')
    # output_csv = os.path.join(config.machine.pipeline_dir, f'mejiro_metadata_{now_string}.csv')

    lens_pickles = sorted(glob(config.machine.dir_02 + '/lens_with_subhalos_*.pkl'))
    lens_uids = [int(os.path.basename(i).split('_')[3].split('.')[0]) for i in lens_pickles]

    pipeline_params = util.hydra_to_dict(config.pipeline)
    bands = pipeline_params['bands']
    limit = pipeline_params['limit']
    pieces = pipeline_params['pieces']
    if limit is not None:
        lens_uids = lens_uids[:limit]

    for uid in tqdm(lens_uids):
        lens = util.unpickle(os.path.join(config.machine.dir_03, f'lens_{str(uid).zfill(8)}.pkl'))
        images = [np.load(f'{config.machine.dir_04}/galsim_{lens.uid}_{band}.npy') for band in bands]
        if pieces:
            # lens_surface_brightness = [np.load(f'{config.machine.dir_03}/array_{lens.uid}_lens_{band}.npy') for band in bands]
            source_surface_brightness = [np.load(f'{config.machine.dir_04}/galsim_{lens.uid}_source_{band}.npy') for
                                         band in bands]
        color_image = np.load(os.path.join(config.machine.dir_05, f'galsim_color_{lens.uid}.npy'))

        fits_path = os.path.join(output_dir, f'strong_lens_{str(uid).zfill(8)}.fits')

        primary_header = fits.Header()

        # mejiro metadata
        import mejiro
        primary_header['VERSION'] = (mejiro.__version__, 'mejiro version')

        import getpass
        import platform
        primary_header['AUTHOR'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')

        # TODO detector and detector position
        # header['DETECTOR']
        # header['DET_X']
        # header['DET_Y']
        # TODO RA and dec

        primary_header['R_BAND'] = (pipeline_params['rgb_bands'][0], 'Filter for red channel of RGB image')
        primary_header['G_BAND'] = (pipeline_params['rgb_bands'][1], 'Filter for green channel of RGB image')
        primary_header['B_BAND'] = (pipeline_params['rgb_bands'][2], 'Filter for blue channel of RGB image')

        primary_hdu = fits.PrimaryHDU(color_image, primary_header)
        hdul = fits.HDUList([primary_hdu])

        for band, array in zip(bands, images):
            header = get_image_header(pipeline_params, band)
            header['EXPOSURE'] = (pipeline_params['exposure_time'], 'Exposure time [seconds]')
            image_hdu = fits.ImageHDU(array, header, name=band)
            hdul.append(image_hdu)

        for band, array in zip(bands, source_surface_brightness):
            header = get_image_header(pipeline_params, band)
            image_hdu = fits.ImageHDU(array, header, name=f'{band} SOURCE')
            hdul.append(image_hdu)

        # for band, array in zip(bands, lens_surface_brightness):
        #     header = get_image_header(pipeline_params, band)
        #     image_hdu = fits.ImageHDU(array, header, name=f'{band} LENS')
        #     hdul.append(image_hdu)

        hdul.writeto(fits_path, overwrite=True)

    # color_image_paths = sorted(glob(f'{config.machine.dir_05}/*.npy'))
    # subhalo_paths = sorted(glob(f'{config.machine.dir_02}/subhalos/*'))
    # image_paths = sorted(glob(f'{config.machine.dir_04}/*[0-9].npy'))
    # lens_image_paths = sorted(glob(f'{config.machine.dir_04}/*_lens.npy'))
    # source_image_paths = sorted(glob(f'{config.machine.dir_04}/*_source.npy'))

    # color_images = [np.load(f) for f in color_image_paths]
    # subhalos = [util.unpickle(f) for f in subhalo_paths]
    # images = [np.load(f) for f in image_paths]
    # lens_images = [np.load(f) for f in lens_image_paths]
    # source_images = [np.load(f) for f in source_image_paths]

    # fits_files = sorted

    # with h5py.File(output_file, 'w') as hf:
    #     # create datasets
    #     image_dataset = hf.create_dataset('images', data=images)
    #     # subhalo_dataset = hf.create_dataset('subhalos', data=subhalos)

    #     # set attributes
    #     hf.attrs['n_images'] = len(images)
    #     # for key, value in util.hydra_to_dict(config.pipeline).items():
    #     #     hf.attrs[key] = value

    # print(f'Output .hdf5 file: {output_file}')

    # # generate csv file
    # lens_paths = glob(f'{config.machine.dir_03}/lens_*')
    # lens_paths.sort()

    # with open(output_csv, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(StrongLens.get_csv_headers())

    #     for lens_path in tqdm(lens_paths):
    #         lens = util.unpickle(lens_path)
    #         writer.writerow(lens.csv_row())

    # print(f'Output .csv file: {output_csv}')

    stop = time.time()
    util.print_execution_time(start, stop)


def get_image_header(pipeline_params, band):
    import mejiro
    import getpass
    import platform

    header = fits.Header()
    header['VERSION'] = (mejiro.__version__, 'mejiro version')
    header['AUTHOR'] = (f'{getpass.getuser()}@{platform.node()}', 'username@host for calculation')
    header['INSTRUME'] = ('WFI', 'Instrument')
    header['FILTER'] = (band, 'Filter')
    # header['EXPOSURE'] = (pipeline_params['exposure_time'], 'Exposure time [seconds]')
    header['OVERSAMP'] = (pipeline_params['grid_oversample'], 'Oversampling used in calculation')
    header['PIXELSCL'] = (0.11, 'Pixel scale [arcsec/pixel]')
    header['FOV'] = (0.11 * pipeline_params['final_pixel_side'], 'Field of view [arcsec]')
    return header


if __name__ == '__main__':
    main()
