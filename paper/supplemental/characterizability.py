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
from scipy.stats import chi2
from tqdm import tqdm


def process_lens(lens, roman, band, scene_size, oversample, kwargs_numerics, instrument_params, psf, rng, exposure_time,
                 snr_quantile, save_dir):
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure
    from mejiro.analysis import stats
    from mejiro.plots import plot_util

    # get image with no subhalos
    lens_no_subhalos = deepcopy(lens)

    synth_no_subhalos = SyntheticImage(lens_no_subhalos,
                                       roman,
                                       band=band,
                                       arcsec=scene_size,
                                       oversample=oversample,
                                       kwargs_numerics=kwargs_numerics,
                                       instrument_params=instrument_params,
                                       pieces=True,
                                       verbose=False)

    engine_params = {
        'rng': rng,
    }
    exposure_no_subhalos = Exposure(synth_no_subhalos,
                                    exposure_time=exposure_time,
                                    engine_params=engine_params,
                                    psf=psf,
                                    verbose=False)

    source_exposure = exposure_no_subhalos.source_exposure

    # get noise
    poisson_noise = exposure_no_subhalos.poisson_noise
    reciprocity_failure = exposure_no_subhalos.reciprocity_failure
    dark_noise = exposure_no_subhalos.dark_noise
    nonlinearity = exposure_no_subhalos.nonlinearity
    ipc = exposure_no_subhalos.ipc
    read_noise = exposure_no_subhalos.read_noise

    # calculate SNR in each pixel
    snr_array = np.nan_to_num(source_exposure / np.sqrt(exposure_no_subhalos.exposure))

    # build SNR mask
    snr_threshold = np.quantile(snr_array, snr_quantile)
    masked_snr_array = np.ma.masked_where(snr_array <= snr_threshold, snr_array)
    mask = np.ma.getmask(masked_snr_array)
    masked_exposure_no_subhalos = np.ma.masked_array(exposure_no_subhalos.exposure, mask=mask)

    # get image with subhalos
    lens_with_subhalos = deepcopy(lens)
    realization = lens_with_subhalos.generate_cdm_subhalos()
    lens_with_subhalos.add_subhalos(realization)

    synth = SyntheticImage(lens_with_subhalos,
                           roman,
                           band=band,
                           arcsec=scene_size,
                           oversample=oversample,
                           kwargs_numerics=kwargs_numerics,
                           instrument_params=instrument_params,
                           verbose=False)

    engine_params.update({
        'poisson_noise': poisson_noise,
        'reciprocity_failure': reciprocity_failure,
        'dark_noise': dark_noise,
        'nonlinearity': nonlinearity,
        'ipc': ipc,
        'read_noise': read_noise
    })

    exposure = Exposure(synth,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf=psf,
                        verbose=False)

    # initialize chi2 rv
    pixels_unmasked = masked_snr_array.count()
    dof = pixels_unmasked - 3
    rv = chi2(dof)
    threshold_chi2 = rv.isf(0.001)

    # mask image with subhalo
    masked_exposure_with_subhalos = np.ma.masked_array(exposure.exposure, mask=mask)

    # calculate chi square
    chi_square = stats.chi_square(np.ma.compressed(masked_exposure_with_subhalos),
                                  np.ma.compressed(masked_exposure_no_subhalos))

    result = {
        "chi_square": chi_square,
        "threshold_chi2": threshold_chi2,
        "uid": lens.uid,
        "synth": synth,
        "synth_no_subhalos": synth_no_subhalos,
        "exposure": exposure,
        "exposure_no_subhalos": exposure_no_subhalos
    }

    if chi_square > threshold_chi2:
        try:
            v_synth = plot_util.get_v(synth.image - synth_no_subhalos.image)
            v_exposure = plot_util.get_v(exposure.exposure - exposure_no_subhalos.exposure)

            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            im1 = ax[0].imshow(synth.image - synth_no_subhalos.image, cmap='bwr', vmin=-v_synth, vmax=v_synth)
            im2 = ax[1].imshow(exposure.exposure - exposure_no_subhalos.exposure, cmap='bwr', vmin=-v_exposure,
                               vmax=v_exposure)
            plt.colorbar(im1, ax=ax[0])
            plt.colorbar(im2, ax=ax[1])
            plt.title(f'{lens.uid} chi2: {chi_square:.2f}, threshold: {threshold_chi2:.2f}')
            plt.savefig(os.path.join(save_dir, f'{lens.uid}_residuals.png'))
            plt.close()
        except Exception as e:
            print(e)

    return result


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses import lens_util
    from mejiro.utils import util
    from mejiro.instruments.roman import Roman
    from mejiro.engines import webbpsf_engine

    pipeline_params = util.hydra_to_dict(config.pipeline)
    debugging = pipeline_params['debugging']

    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    detectable_lenses = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, verbose=True, limit=None)

    # set nice level
    os.nice(pipeline_params['nice'])

    save_dir = os.path.join(config.machine.data_dir, 'characterizability')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    roman = Roman()

    sca = 1
    sca_position = (2048, 2048)
    band = 'F129'
    oversample = 5
    scene_size = 10
    psf_cache_dir = f'{config.machine.data_dir}/cached_psfs'
    rng = galsim.UniformDeviate(42)
    exposure_time = 146
    snr_quantile = 0.95

    instrument_params = {
        'detector': sca,
        'detector_position': sca_position
    }
    kwargs_numerics = {
        'supersampling_factor': 3,
        'compute_mode': 'adaptive',
    }

    psf_id = webbpsf_engine.get_psf_id(band, sca, sca_position, oversample, 101)
    psf = webbpsf_engine.get_cached_psf(psf_id, psf_cache_dir, verbose=False)

    num_characterizable = 0
    chi2_values = []
    uids = []

    count = len(detectable_lenses)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count  # - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [
            executor.submit(process_lens, lens, roman, band, scene_size, oversample, kwargs_numerics, instrument_params,
                            psf, rng, exposure_time, snr_quantile, save_dir) for lens in detectable_lenses]

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()

                chi2_values.append(result["chi_square"])
                uid = result["uid"]
                util.pickle(os.path.join(save_dir, f'result_{uid}.pkl'), result)

                if result["chi_square"] > result["threshold_chi2"]:
                    num_characterizable += 1
                    uids.append(uid)

            except Exception as e:
                print(f"Error encountered: {e}")

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

    execution_time_per_lens = str(datetime.timedelta(seconds=round((stop - start) / len(detectable_lenses))))
    print(f'Execution time per lens: {execution_time_per_lens}')

    print(f'Number of characterizable lenses: {num_characterizable}')
    print(f'Percentage of characterizable lenses: {num_characterizable / len(detectable_lenses) * 100:.2f}%')

    np.save(os.path.join(save_dir, 'chi2_values.npy'), chi2_values)
    np.save(os.path.join(save_dir, 'uids.npy'), uids)


if __name__ == '__main__':
    main()
