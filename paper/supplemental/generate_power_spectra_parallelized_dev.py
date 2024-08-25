import datetime
import multiprocessing
import os
import sys
import time
from copy import deepcopy
from multiprocessing import Pool

import hydra
import matplotlib.pyplot as plt
import numpy as np
from galsim import InterpolatedImage, Image
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm


def get_masked_exposure(lens, model, band, psf, num_pix, oversample, exposure_time, snr_threshold):
    from mejiro.helpers import gs
    from mejiro.helpers import psf as psf_util
    from mejiro.utils import util

    if type(psf) is np.ndarray:
        psf = psf_util.kernel_to_galsim_interpolated_image(psf, oversample)

    total_flux_cps = lens.get_total_flux_cps(band)
    interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / oversample, flux=total_flux_cps * exposure_time)
    image = gs.convolve(interp, psf, num_pix)
    final_array = image.array

    # mask = np.ma.getmask(lens.masked_snr_array)
    data = lens.masked_snr_array.data
    strict_mask = np.ma.masked_where(data < snr_threshold, data)
    mask = np.ma.getmask(strict_mask)
    if mask.shape != final_array.shape:
        mask = util.center_crop_image(mask, final_array.shape)
    masked_image = np.ma.masked_array(final_array, mask)
    np.ma.set_fill_value(masked_image, 0)
    return masked_image.filled()


def get_large(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=9.5, log_mhigh=11., cone_factor=3)


def get_med(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=7.5, log_mhigh=9.5, cone_factor=4)


def get_small(lens, subhalo_params):
    return _get_subhalos(lens, subhalo_params, log_mlow=6., log_mhigh=7.5, cone_factor=6)


def _get_subhalos(lens, subhalo_params, log_mlow, log_mhigh, cone_factor):
    try:
        return CDM(round(lens.z_lens, 2),
                    round(lens.z_source, 2),
                    sigma_sub=subhalo_params['sigma_sub'],
                    log_mlow=log_mlow,
                    log_mhigh=log_mhigh,
                    log_m_host=np.log10(lens.main_halo_mass),
                    r_tidal=subhalo_params['r_tidal'],
                    cone_opening_angle_arcsec=lens.get_einstein_radius() * cone_factor,
                    LOS_normalization=subhalo_params['los_normalization'])
    except Exception as e:
        print(f'StrongLens {lens.uid}: {e}')
        return None


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.helpers import survey_sim, psf
    from mejiro.utils import util

    # script configuration options
    debugging = False
    require_alignment = False
    limit = 100
    snr_threshold = 50.
    snr_pixel_threshold = 1.
    einstein_radius_threshold = 0.
    log_m_host_threshold = 13.  # 13.3

    # set subhalo and imaging params
    subhalo_params = {
        'r_tidal': 0.5,
        'sigma_sub': 0.055,  
        'los_normalization': 0.
    }
    imaging_params = {
        'oversample': 5,
        'num_pix': 45,
        'side': 4.95,
        'control_band': 'F106',
        'exposure_time': 146,
    }
    bands = ['F106']
    # bands = ['F106', 'F129', 'F158', 'F184']
    position_control = {
        '2': (2048, 2048),
    }
    positions = {
        '5': (2048, 2048),
        '11': (2048, 2048),
        '14': (2048, 2048)
        # '9': (4, 4),
        # '17': (4092, 4092)
    }

    # set up directories for script output
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra_parallelized')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    # debugging? set pipeline dir accordingly
    pipeline_params = util.hydra_to_dict(config.pipeline)
    # TODO set debugging here based on yaml file? or manually above?
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    # collect lenses
    print(f'Collecting lenses...')
    lens_list = survey_sim.collect_all_detectable_lenses(os.path.join(pipeline_dir, '01'), suppress_output=False)
    print(f'Collected {len(lens_list)} candidate lens(es).')
    filtered_lenses = []
    num_lenses = 0
    for lens in lens_list:
        if lens.snr > snr_threshold and lens.get_einstein_radius() > einstein_radius_threshold and np.log10(lens.main_halo_mass) > log_m_host_threshold and lens.snr != np.inf:
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

    # read cached PSFs
    cached_psfs = {}
    psf_cache_dir = os.path.join(config.machine.data_dir, 'cached_psfs')
    psf_id_strings = []
    for band in bands:
        for detector, detector_position in {**position_control, **positions}.items():
            psf_id_strings.append(
                psf.get_psf_id_string(band, detector, detector_position, imaging_params['oversample']))

    for id_string in psf_id_strings:
        cached_psfs[id_string] = psf.load_cached_psf(id_string, psf_cache_dir, suppress_output=False)

    tuple_list = [
        (lens, subhalo_params, imaging_params, position_control, positions, cached_psfs, require_alignment, snr_pixel_threshold, save_dir, image_save_dir, debugging) for
        lens in lenses_to_process]

    # split up the lenses into batches based on core count
    count = len(lenses_to_process)
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
    print(f'Processing {len(tuple_list)} lens(es) that satisfy criteria')
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(generate_power_spectra, batch)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

    execution_time_per_lens = str(datetime.timedelta(seconds=round((stop - start) / len(lenses_to_process))))
    print(f'Execution time per lens: {execution_time_per_lens}')


def generate_power_spectra(tuple):
    from mejiro.helpers import gs, psf
    from mejiro.lenses import lens_util
    from mejiro.utils import util

    # unpack tuple
    (lens, subhalo_params, imaging_params, position_control, positions, cached_psfs, require_alignment, snr_pixel_threshold, save_dir, image_save_dir, debugging) = tuple
    control_band = imaging_params['control_band']
    oversample = imaging_params['oversample']
    num_pix = imaging_params['num_pix']
    side = imaging_params['side']
    exposure_time = imaging_params['exposure_time']

    if debugging: print(f'Processing lens {lens.uid}...')

    lens._set_classes()

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

    plt.imshow(np.log10(array), alpha=0.9, cmap='inferno')
    plt.imshow(subhalo_kappa, alpha=0.4, cmap='binary')
    plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
    plt.axis('off')
    plt.title(f'{lens.uid}: Large Population ({len(large.halos)} subhalo(es))')
    try:
        plt.savefig(os.path.join(image_save_dir, f'00_{lens.uid}_large.png'))
    except Exception as e:
        print(e)
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

    plt.imshow(np.log10(array), alpha=0.9, cmap='inferno')
    plt.imshow(subhalo_kappa, alpha=0.4, cmap='binary')
    plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
    plt.axis('off')
    plt.title(f'{lens.uid}: Medium Population ({len(med.halos)} subhalo(es))')
    try:
        plt.savefig(os.path.join(image_save_dir, f'01_{lens.uid}_med.png'))
    except Exception as e:
        print(e)
    plt.close()

    # small subhalos
    small = get_small(lens, subhalo_params)
    if small is None: sys.exit(0)

    subhalo_lens = deepcopy(lens)
    subhalo_lens.add_subhalos(small)
    subhalo_kappa = subhalo_lens.get_subhalo_kappa(num_pix, side)
    array = subhalo_lens.get_array(num_pix, side, control_band)
    image_x, image_y = subhalo_lens.get_image_positions()

    plt.imshow(np.log10(array), alpha=0.75, cmap='inferno')
    plt.imshow(subhalo_kappa, alpha=0.25, cmap='binary')
    plt.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')
    plt.axis('off')
    plt.title(f'{lens.uid}: Small Population ({len(small.halos)} subhalo(es))')
    try:
        plt.savefig(os.path.join(image_save_dir, f'02_{lens.uid}_small.png'))
    except Exception as e:
        print(e)
    plt.close()

    # ---------------------BUILD WDM/MDM/CDM POPULATIONS---------------------
    wdm = deepcopy(large)
    mdm = wdm.join(med)
    cdm = mdm.join(small)

    util.pickle(os.path.join(save_dir, f'wdm_realization_{lens.uid}_{id(wdm)}.pkl'), wdm)
    util.pickle(os.path.join(save_dir, f'mdm_realization_{lens.uid}_{id(mdm)}.pkl'), mdm)
    util.pickle(os.path.join(save_dir, f'cdm_realization_{lens.uid}_{id(cdm)}.pkl'), cdm)

    wdm_lens = deepcopy(lens)
    mdm_lens = deepcopy(lens)
    cdm_lens = deepcopy(lens)

    wdm_lens.add_subhalos(wdm)
    mdm_lens.add_subhalos(mdm)
    cdm_lens.add_subhalos(cdm)

    wdm_kappa = wdm_lens.get_subhalo_kappa(num_pix, side)
    mdm_kappa = mdm_lens.get_subhalo_kappa(num_pix, side)
    cdm_kappa = cdm_lens.get_subhalo_kappa(num_pix, side)

    ps_kappa_wdm, kappa_r = power_spectrum_1d(wdm_kappa)
    ps_kappa_mdm, _ = power_spectrum_1d(mdm_kappa)
    ps_kappa_cdm, _ = power_spectrum_1d(cdm_kappa)
    np.save(os.path.join(save_dir, 'kappa_r.npy'), kappa_r)
    np.save(os.path.join(save_dir, f'ps_kappa_wdm_{lens.uid}_{id(ps_kappa_wdm)}.npy'), ps_kappa_wdm)
    np.save(os.path.join(save_dir, f'ps_kappa_mdm_{lens.uid}_{id(ps_kappa_mdm)}.npy'), ps_kappa_mdm)
    np.save(os.path.join(save_dir, f'ps_kappa_cdm_{lens.uid}_{id(ps_kappa_cdm)}.npy'), ps_kappa_cdm)

    vmin = 0
    vmax = np.max([wdm_kappa, mdm_kappa, cdm_kappa])

    _, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax[0].imshow(wdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)
    ax[1].imshow(mdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)
    ax[2].imshow(cdm_kappa, cmap='binary', vmin=vmin, vmax=vmax)

    ax[0].set_title('WDM')
    ax[1].set_title('MDM')
    ax[2].set_title('CDM')

    for a in ax:
        a.axis('off')
        a.plot(image_x, image_y, 'go', ms=10, markeredgewidth=2, fillstyle='none')

    plt.suptitle(f'{lens.uid}: Convergence Maps and Image Positions')
    try:
        plt.savefig(os.path.join(image_save_dir, f'03_{lens.uid}_convergence_maps.png'))
    except Exception as e:
        print(e)
    plt.close()

    # ---------------------GENERATE SYNTHETIC IMAGES---------------------
    _, _, wdm_array = wdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True)
    _, _, mdm_array = mdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True)
    _, _, cdm_array = cdm_lens.get_array(num_pix * oversample, side, control_band, return_pieces=True)
    # wdm_array = wdm_lens.get_array(num_pix * oversample, side, control_band)
    # mdm_array = mdm_lens.get_array(num_pix * oversample, side, control_band)
    # cdm_array = cdm_lens.get_array(num_pix * oversample, side, control_band)

    wdm_residual = cdm_array - wdm_array
    mdm_residual = cdm_array - mdm_array
    min = np.min([wdm_residual, mdm_residual])
    max = np.max([wdm_residual, mdm_residual])
    vmax = np.max([np.abs(min), np.abs(max)])

    _, ax = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
    ax[0, 0].imshow(wdm_array)
    ax[0, 1].imshow(mdm_array)
    ax[0, 2].imshow(cdm_array)
    ax[1, 0].imshow(wdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax[1, 1].imshow(mdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)

    ax[0, 0].set_title('WDM')
    ax[0, 1].set_title('MDM')
    ax[0, 2].set_title('CDM')

    for a in ax.flatten():
        a.axis('off')

    plt.suptitle(f'{lens.uid}: Synthetic Images Varying Subhalo Population')
    try:
        plt.savefig(os.path.join(image_save_dir, f'04_{lens.uid}_synthetic_images.png'))
    except Exception as e:
        print(e)
    plt.close()

    wdm_ps, r = power_spectrum_1d(wdm_array)
    mdm_ps, _ = power_spectrum_1d(mdm_array)
    cdm_ps, _ = power_spectrum_1d(cdm_array)

    plt.plot(r, cdm_ps - wdm_ps, label='WDM')
    plt.plot(r, cdm_ps - mdm_ps, label='MDM')
    plt.xscale('log')
    plt.title(f'{lens.uid}: Synthetic Image Power Spectra Residuals Varying Subhalo Population')
    plt.legend()
    try:
        plt.savefig(os.path.join(image_save_dir, f'05_{lens.uid}_synthetic_image_power_spectra.png'))
    except Exception as e:
        print(e)
    plt.close()

    # ---------------------GENERATE EXPOSURES VARYING SUBHALO POPULATION---------------------
    subhalos_psf_id_string = psf.get_psf_id_string(band=control_band, detector=2, detector_position=(2048, 2048), oversample=oversample)
    subhalos_psf_kernel = cached_psfs[subhalos_psf_id_string]

    wdm_exposure = get_masked_exposure(wdm_lens, wdm_array, control_band, subhalos_psf_kernel, num_pix, oversample, exposure_time, snr_pixel_threshold)
    mdm_exposure = get_masked_exposure(mdm_lens, mdm_array, control_band, subhalos_psf_kernel, num_pix, oversample, exposure_time, snr_pixel_threshold)
    cdm_exposure = get_masked_exposure(cdm_lens, cdm_array, control_band, subhalos_psf_kernel, num_pix, oversample, exposure_time, snr_pixel_threshold)

    wdm_residual = cdm_exposure - wdm_exposure
    mdm_residual = cdm_exposure - mdm_exposure
    min = np.min([wdm_residual, mdm_residual])
    max = np.max([wdm_residual, mdm_residual])
    vmax = np.max([np.abs(min), np.abs(max)])

    _, ax = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
    ax[0, 0].imshow(wdm_exposure)
    ax[0, 1].imshow(mdm_exposure)
    ax[0, 2].imshow(cdm_exposure)
    ax[1, 0].imshow(wdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax[1, 1].imshow(mdm_residual, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax[0, 0].set_title('WDM')
    ax[0, 1].set_title('MDM')
    ax[0, 2].set_title('CDM')
    for a in ax.flatten():
        a.axis('off')
    plt.suptitle(f'{lens.uid}: Exposures Varying Subhalo Population')
    try:
        plt.savefig(os.path.join(image_save_dir, f'06_{lens.uid}_exposures_varying_subhalos.png'))
    except Exception as e:
        print(e)
    plt.close() 

    wdm_ps, r = power_spectrum_1d(wdm_exposure)
    mdm_ps, _ = power_spectrum_1d(mdm_exposure)
    cdm_ps, _ = power_spectrum_1d(cdm_exposure)

    # save r; should be same for all, since same size (num_pix, num_pix)
    np.save(os.path.join(save_dir, 'r.npy'), r)
    np.save(os.path.join(save_dir, f'{lens.uid}_im_subs_wdm.npy'), wdm_exposure)
    np.save(os.path.join(save_dir, f'{lens.uid}_im_subs_mdm.npy'), mdm_exposure)
    np.save(os.path.join(save_dir, f'{lens.uid}_im_subs_cdm.npy'), cdm_exposure)
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_subs_wdm.npy'), wdm_ps)
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_subs_mdm.npy'), mdm_ps)
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_subs_cdm.npy'), cdm_ps)

    plt.plot(r, cdm_ps - wdm_ps, label='WDM')
    plt.plot(r, cdm_ps - mdm_ps, label='MDM')
    plt.xscale('log')
    plt.title(f'{lens.uid}: Exposure Power Spectra Residuals Varying Subhalo Population')
    plt.legend()
    try:
        plt.savefig(os.path.join(image_save_dir, f'07_{lens.uid}_exposure_power_spectra_varying_subhalos.png'))
    except Exception as e:
        print(e)
    plt.close() 

    # ---------------------GENERATE EXPOSURES VARYING PSFS---------------------
    position_exposures = []
    for detector, detector_position in {**position_control, **positions}.items():
        exposure = get_masked_exposure(cdm_lens, cdm_array, control_band, cached_psfs[psf.get_psf_id_string(control_band, detector, detector_position, oversample)], num_pix, oversample, exposure_time, snr_pixel_threshold)
        position_exposures.append(exposure)

    pos_0_ps, _ = power_spectrum_1d(position_exposures[0])
    pos_1_ps, _ = power_spectrum_1d(position_exposures[1])
    pos_2_ps, _ = power_spectrum_1d(position_exposures[2])
    pos_3_ps, _ = power_spectrum_1d(position_exposures[3])

    np.save(os.path.join(save_dir, f'{lens.uid}_im_pos_{list(position_control.keys())[0]}.npy'), position_exposures[0])
    np.save(os.path.join(save_dir, f'{lens.uid}_im_pos_{list(positions.keys())[0]}.npy'), position_exposures[1])
    np.save(os.path.join(save_dir, f'{lens.uid}_im_pos_{list(positions.keys())[1]}.npy'), position_exposures[2])
    np.save(os.path.join(save_dir, f'{lens.uid}_im_pos_{list(positions.keys())[2]}.npy'), position_exposures[3])
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_pos_control.npy'), pos_0_ps)
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_pos_1.npy'), pos_1_ps)
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_pos_2.npy'), pos_2_ps)
    np.save(os.path.join(save_dir, f'{lens.uid}_ps_pos_3.npy'), pos_3_ps)

    _, ax = plt.subplots(2, 4, figsize=(12, 7), constrained_layout=True)

    ax[0, 0].imshow(position_exposures[0])
    ax[0, 1].imshow(position_exposures[1])
    ax[0, 2].imshow(position_exposures[2])
    ax[0, 3].imshow(position_exposures[3])

    res_1 = position_exposures[0] - position_exposures[1]
    res_2 = position_exposures[0] - position_exposures[2]
    res_3 = position_exposures[0] - position_exposures[3]

    min = np.min([res_1, res_2, res_3])
    max = np.max([res_1, res_2, res_3])
    vmax = np.max([np.abs(min), np.abs(max)])

    ax[1, 0].imshow(res_1, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax[1, 1].imshow(res_2, cmap='bwr', vmin=-vmax, vmax=vmax)
    ax[1, 2].imshow(res_3, cmap='bwr', vmin=-vmax, vmax=vmax)

    for i, (detector, detector_position) in enumerate({**position_control, **positions}.items()):
        ax[0, i].set_title(f'{detector} ({detector_position})')

    for a in ax.flatten():
        a.axis('off')

    plt.suptitle(f'{lens.uid}: Varying PSFs')
    try:
        plt.savefig(os.path.join(image_save_dir, f'08_{lens.uid}_exposures_varying_psfs.png'))
    except Exception as e:
        print(e)
    plt.close() 

    # ---------------------COMPARE POWER SPECTRA RESIDUALS---------------------
    plt.plot(r, pos_0_ps - pos_1_ps, label='SCA04')
    plt.plot(r, pos_0_ps - pos_2_ps, label='SCA02')
    plt.plot(r, pos_0_ps - pos_3_ps, label='SCA05')
    plt.plot(r, cdm_ps - wdm_ps, label='WDM')
    plt.plot(r, cdm_ps - mdm_ps, label='MDM')
    plt.xscale('log')
    plt.title(f'{lens.uid}: Comparing Power Spectra Residuals')
    plt.legend()
    try:
        plt.savefig(os.path.join(image_save_dir, f'09_{lens.uid}_compare_power_spectra_residuals.png'))
    except Exception as e:
        print(e)
    plt.close() 

    print(f'Finished lens {lens.uid}.')


if __name__ == '__main__':
    main()
