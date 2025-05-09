import datetime
import galsim
import hydra
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.single_realization import SingleHalo
from scipy.stats import chi2
from tqdm import tqdm


def process_lens(params):
    from mejiro.analysis import stats
    from mejiro.utils import util
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure
    from mejiro.plots import plot_util
    from mejiro.engines import webbpsf_engine

    (run, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir,
     idx_to_save) = params
    num_positions = script_config['num_positions']
    rng = script_config['rng']
    band = imaging_params['band']
    oversample = imaging_params['oversample']
    scene_size = imaging_params['scene_size']
    exposure_time = imaging_params['exposure_time']
    masses = subhalo_params['masses']
    psf_cache_dir = script_config['psf_cache_dir']

    kwargs_numerics = {
        'supersampling_factor': 3,
        'compute_mode': 'adaptive',
    }

    detectable_halos = []
    results = {}

    # generate and add subhalos
    # realization = lens.generate_cdm_subhalos()
    # lens.add_subhalos(realization)

    # calculate image position and set subhalo position
    image_x, image_y = lens.get_image_positions(pixel_coordinates=False)
    image_distance = np.sqrt(image_x ** 2 + image_y ** 2)
    # more_distant_image_index = np.argmax(image_distance)
    image_index = np.random.choice(len(image_distance))
    halo_x = image_x[image_index]
    halo_y = image_y[image_index]

    for sca, sca_position in positions:
        sca = int(sca)
        instrument_params = {
            'detector': sca,
            'detector_position': sca_position
        }
        position_key = f'{sca}_{sca_position[0]}_{sca_position[1]}'
        results[position_key] = {}

        # get PSF
        psf_id = webbpsf_engine.get_psf_id(band, sca, sca_position, oversample, 101)
        psf = webbpsf_engine.get_cached_psf(psf_id, psf_cache_dir, verbose=False)

        lens_no_subhalo = deepcopy(lens)
        synth_no_subhalo = SyntheticImage(lens_no_subhalo,
                                          roman,
                                          band=band,
                                          arcsec=scene_size,
                                          oversample=oversample,
                                          kwargs_numerics=kwargs_numerics,
                                          instrument_params=instrument_params,
                                          pieces=True,
                                          verbose=False)
        try:
            engine_params = {
                'rng': rng,
                'nonlinearity': False
            }
            exposure_no_subhalo = Exposure(synth_no_subhalo,
                                           exposure_time=exposure_time,
                                           engine_params=engine_params,
                                           psf=psf,
                                           verbose=False)
        except Exception as e:
            print(f'Failed to generate exposure for StrongLens {lens.uid}: {e}')
            return lens.uid, None, None

        source_exposure = exposure_no_subhalo.source_exposure

        poisson_noise = exposure_no_subhalo.poisson_noise
        reciprocity_failure = exposure_no_subhalo.reciprocity_failure
        dark_noise = exposure_no_subhalo.dark_noise
        # nonlinearity = exposure_no_subhalo.nonlinearity
        ipc = exposure_no_subhalo.ipc
        read_noise = exposure_no_subhalo.read_noise

        snr_array = np.nan_to_num(source_exposure / np.sqrt(exposure_no_subhalo.exposure))
        snr_threshold = np.quantile(snr_array, script_config['snr_quantile'])
        masked_snr_array = np.ma.masked_where(snr_array <= snr_threshold, snr_array)
        mask = np.ma.getmask(masked_snr_array)
        masked_exposure_no_subhalo = np.ma.masked_array(exposure_no_subhalo.exposure, mask=mask)

        pixels_unmasked = masked_snr_array.count()
        dof = pixels_unmasked - 3
        rv = chi2(dof)
        threshold_chi2 = rv.isf(0.001)

        for m200 in masses:
            mass_key = f'{int(m200)}'
            results[position_key][mass_key] = []

            chi_square_list = []
            for i in range(num_positions):
                center_x = halo_x
                center_y = halo_y

                pyhalo_lens_cosmo = LensCosmo(lens.z_lens, lens.z_source)
                astropy_class = pyhalo_lens_cosmo.cosmo
                c_model, kwargs_concentration_model = preset_concentration_models('DIEMERJOYCE19')
                kwargs_concentration_model['scatter'] = False
                kwargs_concentration_model['cosmo'] = astropy_class
                concentration_model = c_model(**kwargs_concentration_model)
                truncation_model = None
                kwargs_halo_model = {
                    'truncation_model': truncation_model,
                    'concentration_model': concentration_model,
                    'kwargs_density_profile': {}
                }
                single_halo = SingleHalo(halo_mass=m200,
                                         x=center_x, y=center_y,
                                         mdef='NFW',
                                         z=lens.z_lens, zlens=lens.z_lens, zsource=lens.z_source,
                                         subhalo_flag=True,
                                         kwargs_halo_model=kwargs_halo_model,
                                         astropy_instance=lens.cosmo,
                                         lens_cosmo=pyhalo_lens_cosmo)
                c = single_halo.halos[0].c

                lens_with_subhalo = deepcopy(lens)
                lens_with_subhalo.add_subhalos(single_halo)
                synth = SyntheticImage(lens_with_subhalo,
                                       roman,
                                       band=band,
                                       arcsec=scene_size,
                                       oversample=oversample,
                                       kwargs_numerics=kwargs_numerics,
                                       instrument_params=instrument_params,
                                       verbose=False)
                try:
                    engine_params = {
                        'rng': rng,
                        'poisson_noise': poisson_noise,
                        'dark_noise': dark_noise,
                        'reciprocity_failure': reciprocity_failure,
                        'nonlinearity': False,
                        'ipc': ipc,
                        'read_noise': read_noise
                    }
                    exposure = Exposure(synth,
                                        exposure_time=exposure_time,
                                        engine_params=engine_params,
                                        psf=psf,
                                        verbose=False)
                except Exception as e:
                    print(f'Failed to generate exposure with subhalo for StrongLens {lens.uid}: {e}')
                    return lens.uid, None, None

                masked_exposure_with_subhalo = np.ma.masked_array(exposure.exposure, mask=mask)
                chi_square = stats.chi_square(np.ma.compressed(masked_exposure_with_subhalo),
                                              np.ma.compressed(masked_exposure_no_subhalo))

                if chi_square < 0.:
                    print(f'{lens.uid}: {chi_square=}')
                    return lens.uid, None, None

                chi_square_list.append(chi_square)
                if chi_square > threshold_chi2:
                    detectable_halos.append((lens.z_lens, m200, c))

                if run == idx_to_save:
                    try:
                        _, ax = plt.subplots(2, 3, figsize=(15, 10))
                        residual = masked_exposure_with_subhalo - masked_exposure_no_subhalo
                        vmax = plot_util.get_limit(residual)
                        if vmax == 0.: vmax = 0.1
                        synth.set_native_coords()
                        coords_x, coords_y = synth.coords_native.map_coord2pix(halo_x, halo_y)

                        ax00 = ax[0, 0].imshow(masked_exposure_no_subhalo)
                        ax01 = ax[0, 1].imshow(masked_exposure_with_subhalo)
                        ax[0, 1].scatter(coords_x, coords_y, c='r', s=10)
                        ax02 = ax[0, 2].imshow(residual, cmap='bwr', vmin=-vmax, vmax=vmax)

                        ax[0, 0].set_title(f'SNR: {lens.snr:.2f}')
                        ax[0, 1].set_title(f'{m200:.2e}')
                        ax[0, 2].set_title(
                            r'$\chi^2=$ ' + f'{chi_square:.2f}, ' + r'$\chi_{3\sigma}^2=$ ' + f'{threshold_chi2:.2f}')

                        synth_residual = synth.image - synth_no_subhalo.image
                        vmax_synth = plot_util.get_limit(synth_residual)
                        ax10 = ax[1, 0].imshow(synth_residual, cmap='bwr', vmin=-vmax_synth, vmax=vmax_synth)
                        ax11 = ax[1, 1].imshow(exposure.exposure)
                        ax12 = ax[1, 2].imshow(snr_array)

                        ax[1, 0].set_title('Synthetic Residual')
                        ax[1, 1].set_title('Full Exposure With Subhalo')
                        ax[1, 2].set_title(f'SNR Array: {pixels_unmasked} pixels, {dof} dof')

                        plt.colorbar(ax00, ax=ax[0, 0])
                        plt.colorbar(ax01, ax=ax[0, 1])
                        plt.colorbar(ax02, ax=ax[0, 2])
                        plt.colorbar(ax10, ax=ax[1, 0])
                        plt.colorbar(ax11, ax=ax[1, 1])
                        plt.colorbar(ax12, ax=ax[1, 2])

                        plt.suptitle(
                            f'StrongLens {lens.uid}, {sca_position} on SCA{sca}, Image Shape: {exposure.exposure.shape}')
                        plt.savefig(os.path.join(image_save_dir, f'{lens.uid}_{run}.png'))
                        plt.close()
                    except Exception as e:
                        print(e)

            pvals = [rv.sf(chi) for chi in chi_square_list]
            results[position_key][mass_key] = pvals

    util.pickle(os.path.join(save_dir, f'results_{lens.uid}_{run}.pkl'), results)
    util.pickle(os.path.join(save_dir, f'detectable_halos_{lens.uid}_{run}.pkl'), detectable_halos)

    return lens.uid, results, detectable_halos


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util
    from mejiro.instruments.roman import Roman

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)

    # set nice level
    os.nice(pipeline_params['nice'])

    # script configuration options
    debugging = False
    script_config = {
        'snr_quantile': 0.95,
        'num_lenses': 1000,  # None
        'num_positions': 1,
        'rng': galsim.UniformDeviate(42),
        'psf_cache_dir': os.path.join(config.machine.data_dir, 'cached_psfs')
    }
    subhalo_params = {
        'masses': np.logspace(6, 12, 100),
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'band': 'F106',  # F106
        'scene_size': 5,  # arcsec
        'oversample': 5,
        'exposure_time': 12500  # 438  # 37500  # 12500
    }
    positions = [(1, (2044, 2044))]
    print(f'Processing {len(positions)} positions.')

    save_dir = os.path.join(config.machine.data_dir, 'output', 'subhalo_detection_compare_hltds_wide')  # hlwas  # hltds_deep  # hltds_wide
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir
    print(f'Collecting lenses from {pipeline_dir}')
    lens_list = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, verbose=True)
    og_count = len(lens_list)
    lens_list = [lens for lens in lens_list if lens.snr > 115.47 and lens.snr != np.inf]
    num_unique_systems = len(lens_list)
    num_lenses = script_config['num_lenses']
    if num_lenses is not None:
        repeats = int(np.ceil(num_lenses / len(lens_list)))
        print(f'Repeating lenses {repeats} time(s)')
        lens_list *= repeats
        lens_list = lens_list[:num_lenses]
    print(f'Processing {num_unique_systems} lens(es) of {og_count}')

    num_permutations = len(positions) * len(subhalo_params['masses']) * script_config['num_positions']
    idx_to_save = np.random.randint(low=num_permutations, size=len(lens_list))

    roman = Roman()
    params_list = [
        (
            run, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir,
            idx_to_save[run])
        for run, lens in enumerate(lens_list)]

    count = len(lens_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count  # - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(process_lens, params): params for params in params_list}

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                lens_uid, results, detectable_halos = future.result()
                # if results is not None and detectable_halos is not None:
                #     print(f'Processed StrongLens {lens_uid}')
            except Exception as e:
                print(f"Error encountered: {e}")

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

    execution_time_per_lens = str(datetime.timedelta(seconds=round((stop - start) / len(lens_list))))
    print(f'Execution time per lens: {execution_time_per_lens}')


if __name__ == '__main__':
    main()
