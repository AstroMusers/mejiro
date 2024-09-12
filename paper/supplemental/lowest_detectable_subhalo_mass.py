import datetime
import multiprocessing
import os
import sys
import time
from copy import deepcopy
from multiprocessing import Pool

import galsim
import hydra
import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.Util.correlation import power_spectrum_1d
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util
    from mejiro.instruments.roman import Roman

    # script configuration options
    debugging = True
    script_config = {
        'image_radius': 0.01,  # arcsec
        'num_lenses': 100,
        'num_positions': 1,
        'rng': galsim.UniformDeviate(42)
    }
    subhalo_params = {
        'masses': np.logspace(8, 10, 50),  # 
        'concentration': 6,
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'band': 'F106',
        'scene_size': 5.1,  # arcsec
        'oversample': 1,
        'exposure_time': 146000
    }
    positions = []
    for i in range(1, 19):
        sca = str(i).zfill(2)
        coords = Roman().divide_up_sca(4)
        for coord in coords:
            positions.append((sca, coord))
    print(f'Processing {len(positions)} positions.')

    # set up directories for script output
    save_dir = os.path.join(config.machine.data_dir, 'output', 'lowest_detectable_subhalo_mass')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    # collect lenses
    print(f'Collecting lenses...')
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir
    lens_list = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, suppress_output=False)
    lens_list = [lens for lens in lens_list if lens.snr > 25 and lens.snr < 100]
    num_lenses = script_config['num_lenses']
    print(f'Collected {len(lens_list)} lens(es) and processing {num_lenses}.')
    lens_list = lens_list[:num_lenses]
    # lens_list = np.random.choice(lens_list, num_lenses)
    # lens_list = [lens for lens in lens_list if lens.uid == '00001023' or lens.uid == '00000478']
    print(f'Processing lens(es): {[lens.uid for lens in lens_list]}')

    # print(f'Generating subhalos...')
    # for lens in tqdm(lens_list):
    #     realization = lens.generate_cdm_subhalos()
    #     lens.add_subhalos(realization)
    # print(f'Generated subhalos for {len(lens_list)} lens(es).')

    roman = Roman()
    tuple_list = [
        (
        run, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir, debugging)
        for
        run, lens in enumerate(lens_list)]

    # split up the lenses into batches based on core count
    count = len(lens_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(run, batch)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

    execution_time_per_lens = str(datetime.timedelta(seconds=round((stop - start) / len(lens_list))))
    print(f'Execution time per lens: {execution_time_per_lens}')


def run(tuple):
    from mejiro.analysis import stats
    from mejiro.utils import util
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure

    # unpack tuple
    (_, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir,
     debugging) = tuple
    image_radius = script_config['image_radius']
    num_positions = script_config['num_positions']
    rng = script_config['rng']
    band = imaging_params['band']
    oversample = imaging_params['oversample']
    scene_size = imaging_params['scene_size']
    exposure_time = imaging_params['exposure_time']
    masses = subhalo_params['masses']
    concentration = subhalo_params['concentration']

    # solve for image positions
    image_x, image_y = lens.get_image_positions(pixel_coordinates=False)
    num_images = len(image_x)

    run = 0
    results = {}
    for sca, sca_position in tqdm(positions):
        sca = int(sca)
        position_key = f'{sca}_{sca_position[0]}_{sca_position[1]}'
        # print(' ' + position_key)
        results[position_key] = {}

        # get image with no subhalo
        lens_no_subhalo = deepcopy(lens)
        synth_no_subhalo = SyntheticImage(lens_no_subhalo, 
                                            roman, 
                                            band=band, 
                                            arcsec=scene_size,
                                            oversample=oversample, 
                                            sca=sca, 
                                            sca_position=sca_position,
                                            debugging=False)
        exposure_no_subhalo = Exposure(synth_no_subhalo, 
                                        exposure_time=exposure_time, 
                                        rng=rng, 
                                        sca=sca,
                                        sca_position=sca_position, 
                                        return_noise=True, 
                                        suppress_output=True)

        # get noise
        poisson_noise = exposure_no_subhalo.poisson_noise
        dark_noise = exposure_no_subhalo.dark_noise
        read_noise = exposure_no_subhalo.read_noise

        for m200 in tqdm(masses, disable=True):
            mass_key = f'{int(m200)}'
            results[position_key][mass_key] = []
            # print('     ' + mass_key)

            # compute subhalo parameters
            Rs_angle, alpha_Rs = lens.lens_cosmo.nfw_physical2angle(M=m200, c=concentration)

            chi_square_list = []
            for i in range(num_positions):
                execution_key = f'{position_key}_{mass_key}_{str(i).zfill(8)}'
                # print('         ' + execution_key)

                rand_idx = np.random.randint(0, num_images)  # pick a random image position
                # TODO temp
                halo_x = image_x[0]
                halo_y = image_y[0]
                # halo_x = image_x[rand_idx]
                # halo_y = image_y[rand_idx]

                # get a random point within a small disk around the image
                r = image_radius * np.sqrt(np.random.random_sample())
                theta = 2 * np.pi * np.random.random_sample()
                delta_x, delta_y = util.polar_to_cartesian(r, theta)

                # set subhalo position
                center_x = halo_x + delta_x
                center_y = halo_y + delta_y

                # build subhalo parameters
                subhalo_type = 'TNFW'
                kwargs_subhalo = {
                    'alpha_Rs': alpha_Rs,
                    'Rs': Rs_angle,
                    'center_x': center_x,
                    'center_y': center_y,
                    'r_trunc': 5 * Rs_angle
                }

                # get image with subhalo
                lens_with_subhalo = deepcopy(lens)
                lens_with_subhalo.add_subhalo(subhalo_type, kwargs_subhalo)
                synth = SyntheticImage(lens_with_subhalo, 
                                       roman, 
                                       band=band, 
                                       arcsec=scene_size, 
                                       oversample=oversample,
                                       sca=sca, 
                                       sca_position=sca_position, 
                                       debugging=False)
                exposure = Exposure(synth, 
                                    exposure_time=exposure_time, 
                                    rng=rng, 
                                    sca=sca, 
                                    sca_position=sca_position,
                                    poisson_noise=poisson_noise, 
                                    dark_noise=dark_noise, 
                                    read_noise=read_noise,
                                    suppress_output=True)

                # calculate chi square
                chi_square = stats.chi_square(exposure.exposure, exposure_no_subhalo.exposure)
                chi_square_list.append(chi_square)
                assert chi_square >= 0., print(f'{lens_no_subhalo.uid}: {chi_square=}')
                # print(f'chi square: {chi_square}')

                # save image
                if run % 50 == 0:
                    try:
                        _, ax = plt.subplots(1, 3, figsize=(15, 5))
                        residual = exposure.exposure - exposure_no_subhalo.exposure
                        vmax = np.max(np.abs(residual))
                        if vmax == 0.: vmax = 0.1
                        synth.set_native_coords()
                        coords_x, coords_y = synth.coords_native.map_coord2pix(halo_x, halo_y)
                        ax[0].imshow(exposure_no_subhalo.exposure)
                        ax[0].set_title(f'SNR: {lens.snr:.2f}')
                        ax[1].imshow(exposure.exposure)
                        ax[1].scatter(coords_x, coords_y, c='r', s=10)
                        ax[1].set_title(f'{m200:.2e}')
                        ax[2].imshow(residual, cmap='bwr', vmin=-vmax, vmax=vmax)
                        ax[2].set_title(r'$\chi^2=$ ' + f'{chi_square:.2f}')
                        plt.title(f'{lens.uid}, {exposure.exposure.shape}')
                        plt.savefig(os.path.join(image_save_dir, f'{lens.uid}_{execution_key}.png'))
                        plt.close()
                    except Exception as e:
                        print(e)
                run += 1

            results[position_key][mass_key] = chi_square_list

    util.pickle(os.path.join(save_dir, f'results_{lens.uid}.pkl'), results)


if __name__ == '__main__':
    main()
