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
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import roman_util, util
    from mejiro.lenses import lens_util

    # script configuration options
    debugging = False
    script_config = {
        'num_lenses': 100,  # 1000
        'rng': galsim.UniformDeviate(42),
        'detectors': list(range(1, 19)),
        'detector_positions': roman_util.divide_up_sca(4)
    }
    subhalo_params = {
        'log_mlow': 6,
        'log_mhigh': 10,
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'band': 'F129',
        'scene_size': 5,  # arcsec
        'oversample': 5,
        'exposure_time': 146
    }

    # set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'snrs_across_detectors')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    util.create_directory_if_not_exists(image_save_dir)
    util.create_directory_if_not_exists(os.path.join(image_save_dir, 'snr'))  # for SNR diagnostic plots

    # collect lenses
    print(f'Collecting lenses...')
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir
    lens_list = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, suppress_output=False)
    print(f'Collected {len(lens_list)} candidate lens(es).')
    filtered_lenses = []
    num_lenses = 0
    limit = script_config['num_lenses']
    for lens in lens_list:
        if lens.snr > 100 and lens.snr != np.inf:
            filtered_lenses.append(lens)
            num_lenses += 1
        if limit is not None and num_lenses >= limit:
            break
    print(f'Collected {len(filtered_lenses)} lens(es).')

    # repeat lenses up to limit
    lenses_to_process = []
    if limit is not None:
        repeats = int(np.ceil(limit / len(filtered_lenses)))
        print(f'Repeating lenses {repeats} times...')
        lenses_to_process = filtered_lenses * repeats
        lenses_to_process = lenses_to_process[:limit]

    for i, lens in enumerate(filtered_lenses):
        print(f'{i}: StrongLens {lens.uid}, {np.log10(lens.main_halo_mass):.2f}, {lens.snr:.2f}')

    # tuple the parameters
    tuple_list = [
        (run, lens, roman, script_config, subhalo_params, imaging_params, save_dir, image_save_dir, debugging) for
        run, lens in enumerate(lenses_to_process)]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)  # GalSim needs headroom
    count = limit if limit is not None else len(lenses_to_process)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(run_process, batch)

    stop = time.time()
    _ = util.print_execution_time(start, stop, return_string=True)


def run_process(tuple):
    # a legacy function but prevents duplicate runs
    np.random.seed()

    from mejiro.analysis import regions
    from mejiro.plots import diagnostic_plot
    from mejiro.utils import util
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure

    # unpack tuple
    run, lens, roman, script_config, subhalo_params, imaging_params, save_dir, image_save_dir, debugging = tuple
    detectors = script_config['detectors']
    detector_positions = script_config['detector_positions']
    rng = script_config['rng']
    band = imaging_params['band']
    oversample = imaging_params['oversample']
    scene_size = imaging_params['scene_size']
    exposure_time = imaging_params['exposure_time']

    plot_figures = False
    if run % 10 == 0:
        plot_figures = True
    run = str(run).zfill(3)

    if debugging: print(f'Processing lens {lens.uid}...')
    lens._set_classes()

    # generate and add subhalos
    realization = lens.generate_cdm_subhalos(**subhalo_params)
    lens.add_subhalos(realization)

    results = {}
    for detector in tqdm(detectors, leave=True, disable=False):
        lens_for_noise = deepcopy(lens)
        synth_for_noise = SyntheticImage(lens_for_noise,
                                         roman, band=band, arcsec=scene_size, oversample=oversample, sca=detector,
                                         sca_position=(2048, 2048), debugging=False)
        exposure_for_noise = Exposure(synth_for_noise,
                                      exposure_time=exposure_time, rng=rng, sca=detector, sca_position=(2048, 2048),
                                      return_noise=True, suppress_output=True)

        # NB noise depends on brightnesses of pixels, which in turn depend on SCA-specific zeropoint
        poisson_noise = exposure_for_noise.poisson_noise
        dark_noise = exposure_for_noise.dark_noise
        read_noise = exposure_for_noise.read_noise

        if plot_figures:
            _, ax = plt.subplots(1, 4, figsize=(12, 4))
            ax[0].imshow(exposure_for_noise.exposure)
            ax[0].set_title('Total')
            ax[1].imshow(poisson_noise.array)
            ax[1].set_title(f'Poisson (min: {np.min(poisson_noise.array):.2f})')
            ax[2].imshow(dark_noise.array)
            ax[2].set_title('Dark')
            ax[3].imshow(read_noise.array)
            ax[3].set_title('Read')
            plt.suptitle(f'{lens.uid}: Noise Components')
            try:
                image_save_path = os.path.join(image_save_dir, f'01_{lens.uid}_noise_run{run}.png')
                plt.savefig(image_save_path)
            except Exception as e:
                print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
            plt.close()

        snrs_for_positions = []
        for detector_position in tqdm(detector_positions, leave=False, disable=True):
            lens_tmp = deepcopy(lens)
            synth_tmp = SyntheticImage(lens_tmp,
                                       roman,
                                       band=band,
                                       arcsec=scene_size,
                                       oversample=oversample,
                                       sca=detector,
                                       sca_position=detector_position,
                                       debugging=False,
                                       pieces=True)
            exposure = Exposure(synth_tmp,
                                exposure_time=exposure_time,
                                rng=rng,
                                sca=detector,
                                sca_position=detector_position,
                                poisson_noise=poisson_noise,
                                dark_noise=dark_noise,
                                read_noise=read_noise,
                                suppress_output=True)
            total_exposure = exposure.exposure
            lens_exposure = exposure.lens_exposure
            source_exposure = exposure.source_exposure
            noise = total_exposure - (
                        lens_exposure + source_exposure)  # NB neither lens or source has sky background to detector effects added

            # plt.imshow(source_exposure)
            # plt.colorbar()
            # plt.title(np.min(source_exposure))
            # plt.savefig(os.path.join(image_save_dir, f'02_{lens.uid}_source_exposure_run{run}.png'))
            # plt.close()

            # plt.imshow(total_exposure)
            # plt.colorbar()
            # plt.title(np.min(total_exposure))
            # plt.savefig(os.path.join(image_save_dir, f'02_{lens.uid}_total_exposure_run{run}.png'))
            # plt.close()

            # calculate SNR in each pixel
            np.where(total_exposure <= 0, 1, total_exposure)
            snr_array = np.nan_to_num(source_exposure / np.sqrt(total_exposure), posinf=0., neginf=0.)
            # plt.imshow(snr_array)
            # plt.colorbar()
            # plt.title(np.min(snr_array))
            # plt.savefig(os.path.join(image_save_dir, f'02_{lens.uid}_snr_array_run{run}.png'))
            # plt.close()

            # sys.exit(0)

            # print(np.count_nonzero(np.isnan(snr_array)))
            assert np.any(snr_array >= 1), 'No SNR >= 1'
            assert np.isnan(np.min(snr_array)) == False, 'NaN in SNR array'
            assert np.isinf(np.min(snr_array)) == False, 'Inf in SNR array'
            masked_snr_array = np.ma.masked_where(snr_array <= 1, snr_array)

            # calculate regions of connected pixels given the snr mask
            indices_list = regions.get_regions(masked_snr_array, debug_dir=None)
            if indices_list is None:
                print(f'Error in get_regions for {lens.uid} at {detector_position}')
                continue
            # print(f'{len(indices_list)} regions found')

            snr_list = []
            for region in indices_list:
                numerator, denominator = 0, 0
                for i, j in region:
                    numerator += source_exposure[i, j]
                    denominator += source_exposure[i, j] + lens_exposure[i, j] + noise[i, j]
                if denominator <= 0.:
                    continue
                snr = numerator / np.sqrt(denominator)
                # print(f'{snr=}, {numerator=}, {denominator=}')
                assert not np.isnan(snr), 'NaN in SNR list'
                assert not np.isinf(snr), 'Inf in SNR list'
                snr_list.append(snr)

            snrs_for_positions.append(np.max(snr_list))

        # for last detector position, save a few diagostic plots
        if plot_figures:
            diagnostic_plot.snr_plot(total_exposure, lens_exposure, source_exposure, noise, snr_array, masked_snr_array,
                                     snr_list, image_save_dir)

            _, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(exposure_for_noise.exposure)
            ax[0].set_title('Exposure for noise')
            ax[1].imshow(exposure.exposure)
            ax[1].set_title('Exposure')
            residual = exposure_for_noise.exposure - exposure.exposure
            vmax = np.max(np.abs(residual))
            ax[2].imshow(residual, cmap='bwr', vmin=-vmax, vmax=vmax)
            plt.title(f'{lens.uid}: SNR={np.max(snr_list):.2f}')
            try:
                plt.savefig(os.path.join(image_save_dir, f'02_{lens.uid}_snr_run{run}.png'))
            except Exception as e:
                print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
            plt.close()

        results[detector] = snrs_for_positions

    util.pickle(os.path.join(save_dir, f'results_{lens.uid}.pkl'), results)


if __name__ == '__main__':
    main()
