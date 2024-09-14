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
from scipy.stats import chi2


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
        'snr_quantile': 0.75,
        'image_radius': 0.1,  # arcsec
        'num_lenses': 500,
        'num_positions': 1,
        'rng': galsim.UniformDeviate(42)
    }
    subhalo_params = {
        'masses': np.logspace(7, 8, 100),  # 
        'concentration': 6,
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'band': 'F087',
        'scene_size': 5.61,  # arcsec
        'oversample': 5,  # TODO TEMP
        'exposure_time': 146000000
    }
    positions = []
    for i in range(1, 19):
        sca = str(i).zfill(2)
        coords = Roman().divide_up_sca(4)
        for coord in coords:
            positions.append((sca, coord))
    print(f'Processing {len(positions)} positions.')

    # set up directories for script output
    save_dir = os.path.join(config.machine.data_dir, 'output', 'lowest_detectable_subhalo_mass_dev')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    # collect lenses
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir
    print(f'Collecting lenses from {pipeline_dir}')
    lens_list = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, suppress_output=False)
    og_count = len(lens_list)
    lens_list = [lens for lens in lens_list if lens.snr > 50 and lens.get_einstein_radius() > 0.5]
    num_lenses = script_config['num_lenses']
    if num_lenses is not None:
        repeats = int(np.ceil(num_lenses / len(lens_list)))
        print(f'Repeating lenses {repeats} time(s)')
        lens_list *= repeats
        lens_list = lens_list[:num_lenses]
    print(f'Processing {len(lens_list)} lens(es) of {og_count}')
    # lens_list = np.random.choice(lens_list, num_lenses)
    lens_list = [lens for lens in lens_list if lens.uid != '00000019' and lens.uid != '00000040']
    # lens_list = [lens for lens in lens_list if lens.uid == '00000040']
    # print(f'Processing lens(es): {[lens.uid for lens in lens_list]}')

    # TODO TEMP
    # print(f'Generating subhalos...')
    # for lens in tqdm(lens_list):
    #     realization = lens.generate_cdm_subhalos()
    #     lens.add_subhalos(realization)
    # print(f'Generated subhalos for {len(lens_list)} lens(es).')

    # calculate number of images to save
    num_permutations = len(positions) * len(subhalo_params['masses']) * script_config['num_positions']
    idx_to_save = np.random.randint(low=num_permutations, size=len(lens_list))

    roman = Roman()
    tuple_list = [
        (
        run, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir, idx_to_save[run])
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
     idx_to_save) = tuple
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
                                            pieces=True,
                                            debugging=False)
        try:
            exposure_no_subhalo = Exposure(synth_no_subhalo, 
                                        exposure_time=exposure_time, 
                                        rng=rng, 
                                        sca=sca,
                                        sca_position=sca_position, 
                                        return_noise=True, 
                                        reciprocity_failure=False,
                                        nonlinearity=False,
                                        ipc=False,
                                        suppress_output=True)
        except:
            return
        
        # get pieces
        # lens_exposure = exposure_no_subhalo.lens_exposure
        source_exposure = exposure_no_subhalo.source_exposure

        # get noise
        poisson_noise = exposure_no_subhalo.poisson_noise
        # reciprocity_failure = exposure_no_subhalo.reciprocity_failure
        dark_noise = exposure_no_subhalo.dark_noise
        # nonlinearity = exposure_no_subhalo.nonlinearity
        # ipc = exposure_no_subhalo.ipc
        read_noise = exposure_no_subhalo.read_noise

        # calculate SNR in each pixel
        snr_array = np.nan_to_num(source_exposure / np.sqrt(exposure_no_subhalo.exposure))

        # build SNR mask
        # snr_threshold = np.quantile(snr_array, script_config['snr_quantile'])
        # masked_snr_array = np.ma.masked_where(snr_array <= snr_threshold, snr_array)
        masked_snr_array = util.center_crop_image(lens.masked_snr_array, snr_array.shape)
        mask = np.ma.getmask(masked_snr_array)
        masked_exposure_no_subhalo = np.ma.masked_array(exposure_no_subhalo.exposure, mask=mask)

        # initialize chi2 rv
        pixels_unmasked = masked_snr_array.count()
        dof = pixels_unmasked - 3
        rv = chi2(dof)
        threshold_chi2 = rv.isf(0.001)

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
                try:
                    exposure = Exposure(synth, 
                                    exposure_time=exposure_time, 
                                    rng=rng, 
                                    sca=sca, 
                                    sca_position=sca_position,
                                    poisson_noise=poisson_noise, 
                                    reciprocity_failure=False,
                                    dark_noise=dark_noise, 
                                    nonlinearity=False,
                                    ipc=False,
                                    read_noise=read_noise,
                                    suppress_output=True)
                except:
                    return
                
                # mask image with subhalo
                masked_exposure_with_subhalo = np.ma.masked_array(exposure.exposure, mask=mask)

                # calculate chi square
                chi_square = stats.chi_square(np.ma.compressed(masked_exposure_with_subhalo), np.ma.compressed(masked_exposure_no_subhalo))

                if chi_square < 0.:
                    print(f'{lens_no_subhalo.uid}: {chi_square=}')
                    try:
                        _, ax = plt.subplots(1, 2, figsize=(15, 10))
                        ax0 = ax[0].imshow(masked_exposure_no_subhalo)
                        ax[0].set_title(f'No Subhalo ({np.sum(masked_exposure_no_subhalo < 0)} negative element(s))')
                        ax1 = ax[1].imshow(masked_exposure_with_subhalo)
                        ax[1].set_title(f'With Subhalo ({np.sum(masked_exposure_with_subhalo < 0)} negative element(s))')
                        plt.colorbar(ax0, ax=ax[0])
                        plt.colorbar(ax1, ax=ax[1])
                        plt.savefig(os.path.join(image_save_dir, f'negative_chi2_{lens.uid}_{execution_key}.png'))
                        plt.close()
                    except Exception as e:
                        print(e)
                    return
                chi_square_list.append(chi_square)

                # save image
                if run == idx_to_save:
                    try:
                        _, ax = plt.subplots(2, 3, figsize=(15, 10))
                        residual = exposure.exposure - exposure_no_subhalo.exposure
                        vmax = np.max(np.abs(residual))
                        if vmax == 0.: vmax = 0.1
                        synth.set_native_coords()
                        coords_x, coords_y = synth.coords_native.map_coord2pix(halo_x, halo_y)

                        ax00 = ax[0,0].imshow(masked_exposure_no_subhalo)
                        ax01 = ax[0,1].imshow(masked_exposure_with_subhalo)
                        ax[0,1].scatter(coords_x, coords_y, c='r', s=10)
                        ax02 = ax[0,2].imshow(residual, cmap='bwr', vmin=-vmax, vmax=vmax)

                        ax[0,0].set_title(f'SNR: {lens.snr:.2f}')
                        ax[0,1].set_title(f'{m200:.2e}')
                        ax[0,2].set_title(r'$\chi^2=$ ' + f'{chi_square:.2f}, ' + r'$\chi_{3\sigma}^2=$ ' + f'{threshold_chi2:.2f}')

                        synth_residual = synth.image - synth_no_subhalo.image
                        vmax_synth = np.max(np.abs(synth_residual))
                        ax10 = ax[1,0].imshow(synth_residual, cmap='bwr', vmin=-vmax_synth, vmax=vmax_synth)
                        ax11 = ax[1,1].imshow(exposure.exposure)
                        ax12 = ax[1,2].imshow(snr_array)

                        ax[1,0].set_title('Synthetic Residual')
                        ax[1,1].set_title('Full Exposure With Subhalo')
                        ax[1,2].set_title(f'SNR Array: {pixels_unmasked} pixels, {dof} dof')

                        plt.colorbar(ax00, ax=ax[0,0])
                        plt.colorbar(ax01, ax=ax[0,1])
                        plt.colorbar(ax02, ax=ax[0,2])
                        plt.colorbar(ax10, ax=ax[1,0])
                        plt.colorbar(ax11, ax=ax[1,1])
                        plt.colorbar(ax12, ax=ax[1,2])

                        plt.suptitle(f'StrongLens {lens.uid}, {sca_position} on SCA{sca}, Image Shape: {exposure.exposure.shape}')
                        plt.savefig(os.path.join(image_save_dir, f'{lens.uid}_{execution_key}.png'))
                        plt.close()
                    except Exception as e:
                        print(e)
                run += 1

            # convert chi2 to p-value
            pvals = [rv.sf(chi) for chi in chi_square_list]
            results[position_key][mass_key] = pvals

    util.pickle(os.path.join(save_dir, f'results_{lens.uid}.pkl'), results)


if __name__ == '__main__':
    main()
