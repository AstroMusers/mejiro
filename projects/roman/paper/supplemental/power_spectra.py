import datetime
import hydra
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from galsim import InterpolatedImage, Image
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm


def get_masked_exposure(lens, model, band, psf, num_pix, oversample, exposure_time, snr_threshold, mask=False):
    from mejiro.helpers import gs
    from mejiro.helpers import psf as psf_util
    from mejiro.utils import util

    if type(psf) is np.ndarray:
        psf = psf_util.kernel_to_galsim_interpolated_image(psf, oversample)

    interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / oversample,
                               flux=np.sum(model) * exposure_time)
    image = gs.convolve(interp, psf, num_pix)
    final_array = image.array

    util.center_crop_image(final_array, (90, 90))

    if mask:
        data = lens.masked_snr_array.data
        strict_mask = np.ma.masked_where(data < snr_threshold, data)
        mask = np.ma.getmask(strict_mask)
        if mask.shape != final_array.shape:
            mask = util.center_crop_image(mask, final_array.shape)
        masked_image = np.ma.masked_array(final_array, mask)
        np.ma.set_fill_value(masked_image, 0)
        return masked_image.filled()
    else:
        return final_array


def get_large(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=9., log_mhigh=10., cone_factor=3)


def get_med(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=8., log_mhigh=9., cone_factor=4)


def get_smed(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=7., log_mhigh=8., cone_factor=6)


def get_small(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=6., log_mhigh=7., cone_factor=6)


def _get_subhalos(lens, subhalo_params, log_mlow, log_mhigh, cone_factor):
    try:
        return CDM(round(lens.z_lens, 2),
                   round(lens.z_source, 2),
                   sigma_sub=subhalo_params['sigma_sub'],
                   log_mlow=log_mlow,
                   log_mhigh=log_mhigh,
                   log_m_host=np.log10(lens.main_halo_mass),
                   r_tidal=subhalo_params['r_tidal'],
                   cone_opening_angle_arcsec=5.,
                   LOS_normalization=subhalo_params['los_normalization'])
    except Exception as e:
        print(f'StrongLens {lens.uid}: {e}')
        return None


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    import mejiro
    from mejiro.lenses import lens_util
    from mejiro.helpers import psf
    from mejiro.utils import util

    debugging = False
    require_alignment = False
    limit = None
    snr_threshold = 115.47
    snr_pixel_threshold = 1.
    einstein_radius_threshold = 0.
    log_m_host_threshold = 1

    subhalo_params = {
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'oversample': 5,
        'num_pix': 97,
        'side': 10.67,
        'control_band': 'F106',
        'exposure_time': 438,  # 146
    }
    bands = ['F106']
    position_control = [
        ('2', (2048, 2048))
    ]
    positions = [
        ('1', (2048, 2048)),
        ('2', (2048, 2048)),
        ('3', (2048, 2048)),
        ('4', (2048, 2048)),
        ('5', (2048, 2048)),
        ('6', (2048, 2048)),
        ('7', (2048, 2048)),
        ('8', (2048, 2048)),
        ('9', (2048, 2048)),
        ('10', (2048, 2048)),
        ('11', (2048, 2048)),
        ('12', (2048, 2048)),
        ('13', (2048, 2048)),
        ('14', (2048, 2048)),
        ('15', (2048, 2048)),
        ('16', (2048, 2048)),
        ('17', (2048, 2048)),
        ('18', (2048, 2048))
    ]

    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra_parallelized')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    pipeline_params = util.hydra_to_dict(config.pipeline)
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    module_path = os.path.dirname(mejiro.__file__)
    zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

    print(f'Collecting lenses...')
    lens_list = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, verbose=True)
    print(f'Collected {len(lens_list)} candidate lens(es).')
    filtered_lenses = []
    num_lenses = 0
    for lens in lens_list:
        if lens.snr > snr_threshold and lens.snr != np.inf and lens.get_einstein_radius() > einstein_radius_threshold and np.log10(
                lens.main_halo_mass) > log_m_host_threshold:
            filtered_lenses.append(lens)
            num_lenses += 1
        if limit is not None and num_lenses >= limit:
            break
    print(f'Collected {len(filtered_lenses)} lens(es).')

    lenses_to_process = []
    if limit is not None:
        repeats = int(np.ceil(limit / len(filtered_lenses)))
        print(f'Repeating lenses {repeats} times...')
        lenses_to_process = filtered_lenses * repeats
        lenses_to_process = lenses_to_process[:limit]
    else:
        lenses_to_process = filtered_lenses

    for i, lens in enumerate(lenses_to_process):
        print(f'{i}: StrongLens {lens.uid}, {np.log10(lens.main_halo_mass):.2f}, {lens.snr:.2f}')

    cached_psfs = {}
    psf_cache_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    psf_id_strings = []
    for band in bands:
        for detector, detector_position in position_control + positions:
            psf_id_strings.append(
                psf.get_psf_id_string(band, detector, detector_position, imaging_params['oversample'], 101))

    for id_string in psf_id_strings:
        cached_psfs[id_string] = psf.load_cached_psf(id_string, psf_cache_dir, suppress_output=False)

    tuple_list = [
        (run, lens, subhalo_params, imaging_params, position_control, positions, cached_psfs, require_alignment,
         snr_pixel_threshold, save_dir, image_save_dir, debugging) for
        run, lens in enumerate(lenses_to_process)]

    count = len(lenses_to_process)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    print(f'Processing {len(tuple_list)} lens(es) that satisfy criteria')

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = {executor.submit(generate_power_spectra, batch): batch for batch in tuple_list}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error encountered: {e}")

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

    execution_time_per_lens = str(datetime.timedelta(seconds=round((stop - start) / len(lenses_to_process))))
    print(f'Execution time per lens: {execution_time_per_lens}')


def generate_power_spectra(tuple):
    from mejiro.helpers import psf
    from mejiro.lenses import lens_util
    from mejiro.utils import util

    # unpack tuple
    (run, lens, subhalo_params, imaging_params, position_control, positions, cached_psfs, require_alignment,
     snr_pixel_threshold, save_dir, image_save_dir, debugging) = tuple
    control_band = imaging_params['control_band']
    oversample = imaging_params['oversample']
    num_pix = imaging_params['num_pix']
    side = imaging_params['side']
    exposure_time = imaging_params['exposure_time']

    plot_figures = False
    if run % 10 == 0:
        plot_figures = True
    run = str(run).zfill(3)

    if debugging: print(f'Processing lens {lens.uid}...')
    lens._set_classes()

    from datetime import datetime
    np.random.seed(int(datetime.now().timestamp()))

    # ---------------------GENERATE SUBHALO POPULATIONS---------------------
    # large subhalos
    large_subhalo_exists = False
    while not large_subhalo_exists:
        if require_alignment:
            aligned = False
            i = 0
            while not aligned:
                large = get_large(lens, subhalo_params)
                if large is None: sys.exit(0)
                aligned = lens_util.check_halo_image_alignment(lens, large, halo_mass=5e9)
                i += 1
            print(f'Aligned in {i} iteration(s).')
        else:
            large = get_large(lens, subhalo_params)
        if len(large.halos) > 0:
            large_subhalo_exists = True
    # print(f'{len(large.halos)} subhalo(es) in large subhalo population.')

    subhalo_lens = deepcopy(lens)
    subhalo_lens.add_subhalos(large)
    subhalo_kappa = subhalo_lens.get_subhalo_kappa(num_pix, side)
    array = subhalo_lens.get_array(num_pix, side, control_band)
    image_x, image_y = subhalo_lens.get_image_positions()

    if plot_figures:
        plt.imshow(np.log10(array), alpha=0.9, cmap='inferno')
        plt.imshow(subhalo_kappa, alpha=0.4, cmap='binary')
        plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
        plt.axis('off')
        plt.title(f'{lens.uid}: Large Population ({len(large.halos)} subhalo(es))')
        image_save_path = os.path.join(image_save_dir, f'00_{lens.uid}_large_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # medium subhalos
    medium_subhalo_exists = False
    while not medium_subhalo_exists:
        if require_alignment:
            aligned = False
            i = 0
            while not aligned:
                med = get_med(lens, subhalo_params)
                if med is None: sys.exit(0)
                aligned = lens_util.check_halo_image_alignment(lens, med, halo_mass=5e8)
                i += 1
            print(f'Aligned in {i} iteration(s).')
        else:
            med = get_med(lens, subhalo_params)
        if len(med.halos) > 0:
            medium_subhalo_exists = True
    # print(f'{len(med.halos)} subhalo(es) in medium subhalo population.')

    subhalo_lens = deepcopy(lens)
    subhalo_lens.add_subhalos(med)
    subhalo_kappa = subhalo_lens.get_subhalo_kappa(num_pix, side)
    array = subhalo_lens.get_array(num_pix, side, control_band)
    image_x, image_y = subhalo_lens.get_image_positions()

    if plot_figures:
        plt.imshow(np.log10(array), alpha=0.9, cmap='inferno')
        plt.imshow(subhalo_kappa, alpha=0.4, cmap='binary')
        plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
        plt.axis('off')
        plt.title(f'{lens.uid}: Medium Population ({len(med.halos)} subhalo(es))')
        image_save_path = os.path.join(image_save_dir, f'01_{lens.uid}_med_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # smed subhalos
    smed = get_smed(lens, subhalo_params)
    if smed is None: sys.exit(0)

    subhalo_lens = deepcopy(lens)
    subhalo_lens.add_subhalos(smed)
    subhalo_kappa = subhalo_lens.get_subhalo_kappa(num_pix, side)
    array = subhalo_lens.get_array(num_pix, side, control_band)
    image_x, image_y = subhalo_lens.get_image_positions()

    if plot_figures:
        plt.imshow(np.log10(array), alpha=0.75, cmap='inferno')
        plt.imshow(subhalo_kappa, alpha=0.25, cmap='binary')
        plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
        plt.axis('off')
        plt.title(f'{lens.uid}: Small Population ({len(smed.halos)} subhalo(es))')
        try:
            plt.savefig(os.path.join(image_save_dir, f'02_{lens.uid}_smed_run{run}.png'))
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # small subhalos
    small = get_small(lens, subhalo_params)
    if small is None: sys.exit(0)

    subhalo_lens = deepcopy(lens)
    subhalo_lens.add_subhalos(small)
    subhalo_kappa = subhalo_lens.get_subhalo_kappa(num_pix, side)
    array = subhalo_lens.get_array(num_pix, side, control_band)
    image_x, image_y = subhalo_lens.get_image_positions()

    if plot_figures:
        plt.imshow(np.log10(array), alpha=0.75, cmap='inferno')
        plt.imshow(subhalo_kappa, alpha=0.25, cmap='binary')
        plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
        plt.axis('off')
        plt.title(f'{lens.uid}: Small Population ({len(small.halos)} subhalo(es))')
        try:
            plt.savefig(os.path.join(image_save_dir, f'03_{lens.uid}_small_run{run}.png'))
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # ---------------------BUILD WDM/MDM/CDM POPULATIONS---------------------
    wdm = deepcopy(large)
    mdm = wdm.join(med)
    sdm = mdm.join(smed)
    cdm = sdm.join(small)

    util.pickle(os.path.join(save_dir, f'wdm_realization_{lens.uid}_run{run}.pkl'), wdm)
    util.pickle(os.path.join(save_dir, f'mdm_realization_{lens.uid}_run{run}.pkl'), mdm)
    util.pickle(os.path.join(save_dir, f'sdm_realization_{lens.uid}_run{run}.pkl'), sdm)
    util.pickle(os.path.join(save_dir, f'cdm_realization_{lens.uid}_run{run}.pkl'), cdm)

    wdm_lens = deepcopy(lens)
    mdm_lens = deepcopy(lens)
    sdm_lens = deepcopy(lens)
    cdm_lens = deepcopy(lens)

    wdm_lens.add_subhalos(wdm)
    mdm_lens.add_subhalos(mdm)
    sdm_lens.add_subhalos(sdm)
    cdm_lens.add_subhalos(cdm)

    wdm_kappa = wdm_lens.get_subhalo_kappa(num_pix, side)
    mdm_kappa = mdm_lens.get_subhalo_kappa(num_pix, side)
    sdm_kappa = sdm_lens.get_subhalo_kappa(num_pix, side)
    cdm_kappa = cdm_lens.get_subhalo_kappa(num_pix, side)
    np.save(os.path.join(save_dir, f'im_kappa_wdm_{lens.uid}_run{run}.npy'), wdm_kappa)
    np.save(os.path.join(save_dir, f'im_kappa_mdm_{lens.uid}_run{run}.npy'), mdm_kappa)
    np.save(os.path.join(save_dir, f'im_kappa_sdm_{lens.uid}_run{run}.npy'), sdm_kappa)
    np.save(os.path.join(save_dir, f'im_kappa_cdm_{lens.uid}_run{run}.npy'), cdm_kappa)

    ps_kappa_wdm, kappa_r = power_spectrum_1d(wdm_kappa)
    ps_kappa_mdm, _ = power_spectrum_1d(mdm_kappa)
    ps_kappa_sdm, _ = power_spectrum_1d(sdm_kappa)
    ps_kappa_cdm, _ = power_spectrum_1d(cdm_kappa)
    np.save(os.path.join(save_dir, 'kappa_r.npy'), kappa_r)
    np.save(os.path.join(save_dir, f'ps_kappa_wdm_{lens.uid}_run{run}.npy'), ps_kappa_wdm)
    np.save(os.path.join(save_dir, f'ps_kappa_mdm_{lens.uid}_run{run}.npy'), ps_kappa_mdm)
    np.save(os.path.join(save_dir, f'ps_kappa_sdm_{lens.uid}_run{run}.npy'), ps_kappa_sdm)
    np.save(os.path.join(save_dir, f'ps_kappa_cdm_{lens.uid}_run{run}.npy'), ps_kappa_cdm)

    vmin = 0
    vmax = np.max([wdm_kappa, mdm_kappa, sdm_kappa, cdm_kappa])

    if plot_figures:
        _, ax = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        ax[0].imshow(wdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)
        ax[1].imshow(mdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)
        ax[2].imshow(sdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)
        ax[3].imshow(cdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)

        ax[0].set_title('WDM')
        ax[1].set_title('MDM')
        ax[2].set_title('SDM')
        ax[3].set_title('CDM')

        for a in ax:
            a.axis('off')
            a.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')

        plt.suptitle(f'{lens.uid}: Convergence Maps and Image Positions')
        image_save_path = os.path.join(image_save_dir, f'04_{lens.uid}_convergence_maps_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # ---------------------GENERATE SYNTHETIC IMAGES---------------------
    # build kwargs_numerics
    lens.side = side  # TODO temporary workaround
    lens.num_pix = num_pix * oversample  # TODO temporary workaround
    try:
        supersampling_indices = lens.build_adaptive_grid(num_pix * oversample, pad=25)
    except Exception as e:
        print(f'Error building adaptive grid for {lens.uid}: {e}')
        supersampling_indices = util.create_centered_circle(num_pix * oversample, 3. / (0.11 / 5))
    kwargs_numerics = {
        'supersampling_factor': 3,
        'compute_mode': 'adaptive',
        'supersampled_indexes': supersampling_indices
    }

    # calculate surface brightness arrays
    _, _, wdm_array = wdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True,
                                         kwargs_numerics=kwargs_numerics)
    _, _, mdm_array = mdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True,
                                         kwargs_numerics=kwargs_numerics)
    _, _, sdm_array = sdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True,
                                         kwargs_numerics=kwargs_numerics)
    _, _, cdm_array = cdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True,
                                         kwargs_numerics=kwargs_numerics)

    wdm_residual = wdm_array - cdm_array
    mdm_residual = mdm_array - cdm_array
    sdm_residual = sdm_array - cdm_array
    min = np.min([wdm_residual, mdm_residual, sdm_residual])
    max = np.max([wdm_residual, mdm_residual, sdm_residual])
    vmax = np.max([np.abs(min), np.abs(max)])

    if plot_figures:
        _, ax = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        ax[0, 0].imshow(wdm_array)
        ax[0, 1].imshow(mdm_array)
        ax[0, 2].imshow(sdm_array)
        ax[0, 3].imshow(cdm_array)
        ax[1, 0].imshow(wdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1, 1].imshow(mdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1, 2].imshow(sdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)

        ax[0, 0].set_title('WDM')
        ax[0, 1].set_title('MDM')
        ax[0, 2].set_title('SDM')
        ax[0, 3].set_title('CDM')

        for a in ax.flatten():
            a.axis('off')

        plt.suptitle(f'{lens.uid}: Synthetic Images Varying Subhalo Population')
        image_save_path = os.path.join(image_save_dir, f'04_{lens.uid}_synthetic_images_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    wdm_ps, r = power_spectrum_1d(wdm_array)
    mdm_ps, _ = power_spectrum_1d(mdm_array)
    sdm_ps, _ = power_spectrum_1d(sdm_array)
    cdm_ps, _ = power_spectrum_1d(cdm_array)

    if plot_figures:
        plt.plot(r, wdm_ps - cdm_ps, label='WDM')
        plt.plot(r, mdm_ps - cdm_ps, label='MDM')
        plt.plot(r, sdm_ps - cdm_ps, label='SDM')
        plt.xscale('log')
        plt.title(f'{lens.uid}: Synthetic Image Power Spectra Residuals Varying Subhalo Population')
        plt.legend()
        image_save_path = os.path.join(image_save_dir, f'05_{lens.uid}_synthetic_image_power_spectra_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # ---------------------GENERATE EXPOSURES VARYING SUBHALO POPULATION---------------------
    subhalos_psf_id_string = psf.get_psf_id_string(band=control_band, detector=position_control[0][0],
                                                   detector_position=position_control[0][1], oversample=oversample,
                                                   num_pix=101)
    subhalos_psf_kernel = cached_psfs[subhalos_psf_id_string]

    wdm_exposure = get_masked_exposure(wdm_lens, wdm_array, control_band, subhalos_psf_kernel, num_pix, oversample,
                                       exposure_time, snr_pixel_threshold)
    mdm_exposure = get_masked_exposure(mdm_lens, mdm_array, control_band, subhalos_psf_kernel, num_pix, oversample,
                                       exposure_time, snr_pixel_threshold)
    sdm_exposure = get_masked_exposure(sdm_lens, sdm_array, control_band, subhalos_psf_kernel, num_pix, oversample,
                                       exposure_time, snr_pixel_threshold)
    cdm_exposure = get_masked_exposure(cdm_lens, cdm_array, control_band, subhalos_psf_kernel, num_pix, oversample,
                                       exposure_time, snr_pixel_threshold)
    np.save(os.path.join(save_dir, f'im_wdm_{lens.uid}_run{run}.npy'), wdm_exposure)
    np.save(os.path.join(save_dir, f'im_mdm_{lens.uid}_run{run}.npy'), mdm_exposure)
    np.save(os.path.join(save_dir, f'im_sdm_{lens.uid}_run{run}.npy'), sdm_exposure)
    np.save(os.path.join(save_dir, f'im_cdm_{lens.uid}_run{run}.npy'), cdm_exposure)

    wdm_residual = cdm_exposure - wdm_exposure
    mdm_residual = cdm_exposure - mdm_exposure
    sdm_residual = cdm_exposure - sdm_exposure
    min = np.min([wdm_residual, mdm_residual, sdm_residual])
    max = np.max([wdm_residual, mdm_residual, sdm_residual])
    vmax = np.max([np.abs(min), np.abs(max)])

    if plot_figures:
        _, ax = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        ax[0, 0].imshow(wdm_exposure)
        ax[0, 1].imshow(mdm_exposure)
        ax[0, 2].imshow(sdm_exposure)
        ax[0, 3].imshow(cdm_exposure)
        ax[1, 0].imshow(wdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1, 1].imshow(mdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1, 2].imshow(sdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[0, 0].set_title('WDM')
        ax[0, 1].set_title('MDM')
        ax[0, 2].set_title('SDM')
        ax[0, 3].set_title('CDM')
        for a in ax.flatten():
            a.axis('off')
        plt.suptitle(f'{lens.uid}: Exposures Varying Subhalo Population')
        image_save_path = os.path.join(image_save_dir, f'06_{lens.uid}_exposures_varying_subhalos_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    wdm_ps, r = power_spectrum_1d(wdm_exposure)
    mdm_ps, _ = power_spectrum_1d(mdm_exposure)
    sdm_ps, _ = power_spectrum_1d(sdm_exposure)
    cdm_ps, _ = power_spectrum_1d(cdm_exposure)

    # save r; should be same for all, since same size (num_pix, num_pix)
    np.save(os.path.join(save_dir, 'r.npy'), r)
    # np.save(os.path.join(save_dir, f'{lens.uid}_im_subs_wdm_{id(wdm_exposure)}.npy'), wdm_exposure)
    # np.save(os.path.join(save_dir, f'{lens.uid}_im_subs_mdm_{id(mdm_exposure)}.npy'), mdm_exposure)
    # np.save(os.path.join(save_dir, f'{lens.uid}_im_subs_cdm_{id(cdm_exposure)}.npy'), cdm_exposure)
    # np.save(os.path.join(save_dir, f'{lens.uid}_ps_subs_wdm_{id(wdm_ps)}.npy'), wdm_ps)
    # np.save(os.path.join(save_dir, f'{lens.uid}_ps_subs_mdm_{id(mdm_ps)}.npy'), mdm_ps)
    # np.save(os.path.join(save_dir, f'{lens.uid}_ps_subs_cdm_{id(cdm_ps)}.npy'), cdm_ps)

    res_ps_wdm = cdm_ps - wdm_ps
    res_ps_mdm = cdm_ps - mdm_ps
    res_ps_sdm = cdm_ps - sdm_ps
    np.save(os.path.join(save_dir, f'res_ps_wdm_{lens.uid}_run{run}.npy'), res_ps_wdm)
    np.save(os.path.join(save_dir, f'res_ps_mdm_{lens.uid}_run{run}.npy'), res_ps_mdm)
    np.save(os.path.join(save_dir, f'res_ps_sdm_{lens.uid}_run{run}.npy'), res_ps_sdm)

    if plot_figures:
        plt.plot(r, res_ps_wdm, label='WDM')
        plt.plot(r, res_ps_mdm, label='MDM')
        plt.plot(r, res_ps_sdm, label='SDM')
        plt.xscale('log')
        plt.title(f'{lens.uid}: Exposure Power Spectra Residuals Varying Subhalo Population')
        plt.legend()
        image_save_path = os.path.join(image_save_dir,
                                       f'07_{lens.uid}_exposure_power_spectra_varying_subhalos_run{run}.png')
        try:
            plt.savefig(image_save_path)
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

        # ---------------------GENERATE EXPOSURES VARYING PSFS---------------------
    position_exposures = []
    for detector, detector_position in position_control + positions:
        exposure = get_masked_exposure(cdm_lens, cdm_array, control_band, cached_psfs[
            psf.get_psf_id_string(control_band, detector, detector_position, oversample, 101)], num_pix, oversample,
                                       exposure_time, snr_pixel_threshold)
        position_exposures.append(exposure)
        np.save(os.path.join(save_dir, f'im_pos_{detector}_{lens.uid}_run{run}.npy'), exposure)

    power_spectra_list = []
    for i, position_exposure in enumerate(position_exposures):
        ps, _ = power_spectrum_1d(position_exposure)
        power_spectra_list.append(ps)

    res_im_pos_1 = position_exposures[1] - position_exposures[0]
    res_im_pos_2 = position_exposures[2] - position_exposures[0]
    res_im_pos_3 = position_exposures[3] - position_exposures[0]
    res_im_pos_4 = position_exposures[4] - position_exposures[0]
    res_im_pos_5 = position_exposures[5] - position_exposures[0]
    res_im_pos_6 = position_exposures[6] - position_exposures[0]
    res_im_pos_7 = position_exposures[7] - position_exposures[0]
    res_im_pos_8 = position_exposures[8] - position_exposures[0]
    res_im_pos_9 = position_exposures[9] - position_exposures[0]
    res_im_pos_10 = position_exposures[10] - position_exposures[0]
    res_im_pos_11 = position_exposures[11] - position_exposures[0]
    res_im_pos_12 = position_exposures[12] - position_exposures[0]
    res_im_pos_13 = position_exposures[13] - position_exposures[0]
    res_im_pos_14 = position_exposures[14] - position_exposures[0]
    res_im_pos_15 = position_exposures[15] - position_exposures[0]
    res_im_pos_16 = position_exposures[16] - position_exposures[0]
    res_im_pos_17 = position_exposures[17] - position_exposures[0]
    res_im_pos_18 = position_exposures[18] - position_exposures[0]
    np.save(os.path.join(save_dir, f'res_im_pos_1_{lens.uid}_run{run}.npy'), res_im_pos_1)
    np.save(os.path.join(save_dir, f'res_im_pos_2_{lens.uid}_run{run}.npy'), res_im_pos_2)
    np.save(os.path.join(save_dir, f'res_im_pos_3_{lens.uid}_run{run}.npy'), res_im_pos_3)
    np.save(os.path.join(save_dir, f'res_im_pos_4_{lens.uid}_run{run}.npy'), res_im_pos_4)
    np.save(os.path.join(save_dir, f'res_im_pos_5_{lens.uid}_run{run}.npy'), res_im_pos_5)
    np.save(os.path.join(save_dir, f'res_im_pos_6_{lens.uid}_run{run}.npy'), res_im_pos_6)
    np.save(os.path.join(save_dir, f'res_im_pos_7_{lens.uid}_run{run}.npy'), res_im_pos_7)
    np.save(os.path.join(save_dir, f'res_im_pos_8_{lens.uid}_run{run}.npy'), res_im_pos_8)
    np.save(os.path.join(save_dir, f'res_im_pos_9_{lens.uid}_run{run}.npy'), res_im_pos_9)
    np.save(os.path.join(save_dir, f'res_im_pos_10_{lens.uid}_run{run}.npy'), res_im_pos_10)
    np.save(os.path.join(save_dir, f'res_im_pos_11_{lens.uid}_run{run}.npy'), res_im_pos_11)
    np.save(os.path.join(save_dir, f'res_im_pos_12_{lens.uid}_run{run}.npy'), res_im_pos_12)
    np.save(os.path.join(save_dir, f'res_im_pos_13_{lens.uid}_run{run}.npy'), res_im_pos_13)
    np.save(os.path.join(save_dir, f'res_im_pos_14_{lens.uid}_run{run}.npy'), res_im_pos_14)
    np.save(os.path.join(save_dir, f'res_im_pos_15_{lens.uid}_run{run}.npy'), res_im_pos_15)
    np.save(os.path.join(save_dir, f'res_im_pos_16_{lens.uid}_run{run}.npy'), res_im_pos_16)
    np.save(os.path.join(save_dir, f'res_im_pos_17_{lens.uid}_run{run}.npy'), res_im_pos_17)
    np.save(os.path.join(save_dir, f'res_im_pos_18_{lens.uid}_run{run}.npy'), res_im_pos_18)

    res_ps_pos_1 = power_spectra_list[1] - power_spectra_list[0]
    res_ps_pos_2 = power_spectra_list[2] - power_spectra_list[0]
    res_ps_pos_3 = power_spectra_list[3] - power_spectra_list[0]
    res_ps_pos_4 = power_spectra_list[4] - power_spectra_list[0]
    res_ps_pos_5 = power_spectra_list[5] - power_spectra_list[0]
    res_ps_pos_6 = power_spectra_list[6] - power_spectra_list[0]
    res_ps_pos_7 = power_spectra_list[7] - power_spectra_list[0]
    res_ps_pos_8 = power_spectra_list[8] - power_spectra_list[0]
    res_ps_pos_9 = power_spectra_list[9] - power_spectra_list[0]
    res_ps_pos_10 = power_spectra_list[10] - power_spectra_list[0]
    res_ps_pos_11 = power_spectra_list[11] - power_spectra_list[0]
    res_ps_pos_12 = power_spectra_list[12] - power_spectra_list[0]
    res_ps_pos_13 = power_spectra_list[13] - power_spectra_list[0]
    res_ps_pos_14 = power_spectra_list[14] - power_spectra_list[0]
    res_ps_pos_15 = power_spectra_list[15] - power_spectra_list[0]
    res_ps_pos_16 = power_spectra_list[16] - power_spectra_list[0]
    res_ps_pos_17 = power_spectra_list[17] - power_spectra_list[0]
    res_ps_pos_18 = power_spectra_list[18] - power_spectra_list[0]
    np.save(os.path.join(save_dir, f'res_ps_pos_1_{lens.uid}_run{run}.npy'), res_ps_pos_1)
    np.save(os.path.join(save_dir, f'res_ps_pos_2_{lens.uid}_run{run}.npy'), res_ps_pos_2)
    np.save(os.path.join(save_dir, f'res_ps_pos_3_{lens.uid}_run{run}.npy'), res_ps_pos_3)
    np.save(os.path.join(save_dir, f'res_ps_pos_4_{lens.uid}_run{run}.npy'), res_ps_pos_4)
    np.save(os.path.join(save_dir, f'res_ps_pos_5_{lens.uid}_run{run}.npy'), res_ps_pos_5)
    np.save(os.path.join(save_dir, f'res_ps_pos_6_{lens.uid}_run{run}.npy'), res_ps_pos_6)
    np.save(os.path.join(save_dir, f'res_ps_pos_7_{lens.uid}_run{run}.npy'), res_ps_pos_7)
    np.save(os.path.join(save_dir, f'res_ps_pos_8_{lens.uid}_run{run}.npy'), res_ps_pos_8)
    np.save(os.path.join(save_dir, f'res_ps_pos_9_{lens.uid}_run{run}.npy'), res_ps_pos_9)
    np.save(os.path.join(save_dir, f'res_ps_pos_10_{lens.uid}_run{run}.npy'), res_ps_pos_10)
    np.save(os.path.join(save_dir, f'res_ps_pos_11_{lens.uid}_run{run}.npy'), res_ps_pos_11)
    np.save(os.path.join(save_dir, f'res_ps_pos_12_{lens.uid}_run{run}.npy'), res_ps_pos_12)
    np.save(os.path.join(save_dir, f'res_ps_pos_13_{lens.uid}_run{run}.npy'), res_ps_pos_13)
    np.save(os.path.join(save_dir, f'res_ps_pos_14_{lens.uid}_run{run}.npy'), res_ps_pos_14)
    np.save(os.path.join(save_dir, f'res_ps_pos_15_{lens.uid}_run{run}.npy'), res_ps_pos_15)
    np.save(os.path.join(save_dir, f'res_ps_pos_16_{lens.uid}_run{run}.npy'), res_ps_pos_16)
    np.save(os.path.join(save_dir, f'res_ps_pos_17_{lens.uid}_run{run}.npy'), res_ps_pos_17)
    np.save(os.path.join(save_dir, f'res_ps_pos_18_{lens.uid}_run{run}.npy'), res_ps_pos_18)

    if plot_figures:
        _, ax = plt.subplots(2, 4, figsize=(12, 7), constrained_layout=True)

        ax[0, 0].imshow(position_exposures[0])
        ax[0, 1].imshow(position_exposures[1])
        ax[0, 2].imshow(position_exposures[2])
        ax[0, 3].imshow(position_exposures[3])

        min = np.min([res_im_pos_1, res_im_pos_2, res_im_pos_3])
        max = np.max([res_im_pos_1, res_im_pos_2, res_im_pos_3])
        vmax = np.max([np.abs(min), np.abs(max)])

        ax[1, 1].imshow(res_im_pos_1, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1, 2].imshow(res_im_pos_2, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[1, 3].imshow(res_im_pos_3, cmap='bwr', vmin=-vmax, vmax=vmax)

        # for i, (detector, detector_position) in enumerate(position_control.items()):
        #     ax[0, i].set_title(f'{detector} ({detector_position})')

        for a in ax.flatten():
            a.axis('off')

        plt.suptitle(f'{lens.uid}: Varying PSFs')
        image_save_path = os.path.join(image_save_dir, f'08_{lens.uid}_exposures_varying_psfs_run{run}.png')
        try:
            plt.savefig(os.path.join(image_save_dir, f'08_{lens.uid}_exposures_varying_psfs_run{run}.png'))
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

        # ---------------------COMPARE POWER SPECTRA RESIDUALS---------------------
    if plot_figures:
        plt.plot(r, res_ps_pos_1)  # , label=f'{list(positions.keys())[0]}'
        plt.plot(r, res_ps_pos_2)
        plt.plot(r, res_ps_pos_3)
        plt.plot(r, res_ps_wdm, label='WDM')
        plt.plot(r, res_ps_mdm, label='MDM')
        plt.plot(r, res_ps_sdm, label='SDM')
        plt.xscale('log')
        plt.title(f'{lens.uid}: Comparing Power Spectra Residuals')
        plt.legend()
        image_save_path = os.path.join(image_save_dir, f'09_{lens.uid}_compare_power_spectra_residuals_run{run}.png')
        try:
            plt.savefig(os.path.join(image_save_dir, f'09_{lens.uid}_compare_power_spectra_residuals_run{run}.png'))
        except Exception as e:
            print(f'Failed to save {os.path.basename(image_save_path)}: {e}')
        plt.close()

    # print(f'Finished lens {lens.uid}.')


if __name__ == '__main__':
    main()
