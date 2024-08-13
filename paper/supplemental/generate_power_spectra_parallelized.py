import datetime
import multiprocessing
import os
import sys
import time
from copy import deepcopy
from glob import glob
from multiprocessing import Pool

import hydra
import matplotlib.pyplot as plt
import numpy as np
from galsim import InterpolatedImage, Image
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm


def get_psf_id_string(band, detector, detector_position, oversample):
    return f'{band}_{detector}_{detector_position[0]}_{detector_position[1]}_{oversample}'


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    import mejiro
    from mejiro.helpers import survey_sim
    from mejiro.utils import util

    # script configuration options
    debugging = True
    require_alignment = False
    limit = 100
    snr_threshold = 100

    # set subhalo and imaging params
    subhalo_params = {
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0
    }
    imaging_params = {
        'control_band': 'F129',
        'oversample': 5,  # TODO maybe need to update
        'num_pix': 91,  # 45
        'side': 10.01  # 4.95
    }
    bands = ['F106', 'F129', 'F158', 'F184']
    detectors = [4, 1, 9, 17]
    detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

    # set top directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra_parallelized')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    # collect lenses
    print(f'Collecting lenses...')
    lens_list = survey_sim.collect_all_detectable_lenses(config.machine.dir_01)
    print(f'Collected {len(lens_list)} candidate lens(es).')
    lenses_to_process = []
    if limit is not None:
        num_lenses = 0
        for lens in lens_list:
            if num_lenses < limit:
                if lens.snr > snr_threshold:
                    lenses_to_process.append(lens)
                    num_lenses += 1
            else:
                break
    else:
        lenses_to_process = lens_list
    print(f'Collected {len(lenses_to_process)} lens(es).')

    # read cached PSFs
    cached_psfs = {}
    module_path = os.path.dirname(mejiro.__file__)
    psf_cache_dir = os.path.join(module_path, 'data', 'cached_psfs')
    psf_id_strings = []
    for band in bands:
        for detector, detector_position in zip(detectors, detector_positions):
            psf_id_strings.append(get_psf_id_string(band, detector, detector_position, imaging_params['oversample']))
        
    for id_string in psf_id_strings:
        psf_path = glob(os.path.join(psf_cache_dir, f'{id_string}.pkl'))
        assert len(psf_path) == 1, f'Cached PSF for {id_string} not found in {psf_cache_dir}'
        print(f'Loading cached PSF: {psf_path[0]}')
        cached_psfs[id_string] = util.unpickle(psf_path[0])

    tuple_list = [(lens, subhalo_params, imaging_params, cached_psfs, require_alignment, save_dir, image_save_dir, debugging) for lens in lenses_to_process]

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
    print(f'Processing {len(tuple_list)} lens(es) with SNR > {snr_threshold}')
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
    from mejiro.plots import diagnostic_plot, plot_util
    from mejiro.utils import util

    # unpack tuple
    (lens, subhalo_params, imaging_params, cached_psfs, require_alignment, save_dir, image_save_dir, debugging) = tuple
    r_tidal = subhalo_params['r_tidal']
    sigma_sub = subhalo_params['sigma_sub']
    los_normalization = subhalo_params['los_normalization']
    control_band = imaging_params['control_band']
    oversample = imaging_params['oversample']
    num_pix = imaging_params['num_pix']
    side = imaging_params['side']

    if debugging: print(f'Processing lens {lens.uid}...')

    lens._set_classes()

    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)
    log_m_host = np.log10(lens.main_halo_mass)
    einstein_radius = lens.get_einstein_radius()

    cut_8_success, med_success, smol_success = False, False, False

    if require_alignment:
        cut_8_good = False
        i = 0

        while not cut_8_good:
            while not cut_8_success:
                try:
                    cut_8 = CDM(z_lens,
                                z_source,
                                sigma_sub=sigma_sub,
                                log_mlow=8.,
                                log_mhigh=10.,
                                log_m_host=log_m_host,
                                r_tidal=r_tidal,
                                cone_opening_angle_arcsec=einstein_radius * 3,
                                LOS_normalization=los_normalization)
                    cut_8_good = lens_util.check_halo_image_alignment(lens, cut_8, halo_mass=1e8, halo_sort_massive_first=True, return_halo=False)
                    i += 1
                    cut_8_success = True
                except:
                    cut_8_success = False
        if debugging: print(f'lens {lens.uid}: Generated cut_8 population after {i} iterations.')
    else:
        while not cut_8_success:
            try:
                cut_8 = CDM(z_lens,
                            z_source,
                            sigma_sub=sigma_sub,
                            log_mlow=8.,
                            log_mhigh=10.,
                            log_m_host=log_m_host,
                            r_tidal=r_tidal,
                            cone_opening_angle_arcsec=einstein_radius * 3,
                            LOS_normalization=los_normalization)
                cut_8_success = True
            except:
                cut_8_success = False
    if debugging: print(f'lens {lens.uid}: Generated large population.')

    while not med_success:
        try:
            med = CDM(z_lens,
                    z_source,
                    sigma_sub=sigma_sub,
                    log_mlow=7.,
                    log_mhigh=8.,
                    log_m_host=log_m_host,
                    r_tidal=r_tidal,
                    cone_opening_angle_arcsec=einstein_radius * 3,
                    LOS_normalization=los_normalization)
            med_success = True
        except:
            med_success = False
    if debugging: print(f'lens {lens.uid}: Generated med population.')
    
    while not smol_success:
        try:
            smol = CDM(z_lens,
                    z_source,
                    sigma_sub=sigma_sub,
                    log_mlow=6.,
                    log_mhigh=7.,
                    log_m_host=log_m_host,
                    r_tidal=r_tidal,
                    cone_opening_angle_arcsec=einstein_radius * 3,
                    LOS_normalization=los_normalization)
            smol_success = True
        except:
            smol_success = False
    if debugging: print(f'lens {lens.uid}: Generated smol population.')

    cut_7 = cut_8.join(med)
    cut_6 = cut_7.join(smol)

    util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_8.pkl'), cut_8)
    util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_7.pkl'), cut_7)
    util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_6.pkl'), cut_6)

    lens_cut_6 = deepcopy(lens)
    lens_cut_7 = deepcopy(lens)
    lens_cut_8 = deepcopy(lens)

    lens_cut_6.add_subhalos(cut_6)
    lens_cut_7.add_subhalos(cut_7)
    lens_cut_8.add_subhalos(cut_8)

    lenses = [lens_cut_6, lens_cut_7, lens_cut_8, lens]
    titles = [f'cut_6_{lens.uid}', f'cut_7_{lens.uid}',
                f'cut_8_{lens.uid}', f'no_subhalos_{lens.uid}']
    models = [l.get_array(num_pix=num_pix * oversample, side=side, band=control_band) for l in lenses]

    # set aside the CDM lens and model
    control_lens = deepcopy(lenses[2])
    control_model = deepcopy(models[2])

    # check that the models were generated correctly
    diagnostic_plot.power_spectrum_check(models, lenses, titles, os.path.join(image_save_dir, f'{lens.uid}_00_models.png'), oversampled=True)

    # generate power spectra of convergence maps
    for model, lens, title in zip(models, lenses, titles):
        if title == f'no_subhalos_{lens.uid}':
            kappa = lens.get_macrolens_kappa(num_pix=num_pix * oversample, side=side)
        else:
            kappa = lens.get_total_kappa(num_pix=num_pix * oversample, side=side)
        ps_kappa, kappa_r = power_spectrum_1d(kappa)
        np.save(os.path.join(save_dir, f'kappa_im_{title}.npy'), kappa)
        np.save(os.path.join(save_dir, f'kappa_ps_{title}.npy'), ps_kappa)

        proj_mass = lens.lens_cosmo.kappa2proj_mass(kappa)
        ps_proj_mass, proj_mass_r = power_spectrum_1d(proj_mass)
        np.save(os.path.join(save_dir, f'kappa_im_{title}.npy'), proj_mass)
        np.save(os.path.join(save_dir, f'kappa_ps_{title}.npy'), ps_proj_mass)
    np.save(os.path.join(save_dir, 'kappa_r.npy'), kappa_r)
    np.save(os.path.join(save_dir, 'proj_mass_r.npy'), proj_mass_r)

    # SUBHALO-DEPENDENCE
    subhalos_psf_id_string = get_psf_id_string(band=control_band, detector=1, detector_position=(2048, 2048), oversample=oversample)
    subhalos_psf_kernel = cached_psfs[subhalos_psf_id_string]
    subhalo_images = []
    for model, lens, title in zip(models, lenses, titles):
        total_flux_cps = lens.get_total_flux_cps(control_band)
        interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / oversample, flux=total_flux_cps * 146)
        image = gs.convolve(interp, subhalos_psf_kernel, num_pix)
        final_array = image.array
        subhalo_images.append(final_array)

        ps, r = power_spectrum_1d(final_array)
        np.save(os.path.join(save_dir, f'im_subs_{title}.npy'), final_array)
        np.save(os.path.join(save_dir, f'ps_subs_{title}.npy'), ps)

    diagnostic_plot.power_spectrum_check(subhalo_images, lenses, titles, os.path.join(image_save_dir, f'{lens.uid}_01_subhalos.png'), oversampled=False)

    # POSITION-DEPENDENCE
    detectors = [4, 1, 9, 17]
    detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]
    position_lens = deepcopy(control_lens)
    position_model = deepcopy(control_model)
    position_total_flux_cps = position_lens.get_total_flux_cps(control_band)
    position_interp = InterpolatedImage(Image(position_model, xmin=0, ymin=0), scale=0.11 / oversample, flux=position_total_flux_cps * 146)
    position_images, position_lenses = [], []
    for detector, detector_pos in zip(detectors, detector_positions):
        psf_id_string = get_psf_id_string(control_band, detector, detector_pos, oversample)
        webbpsf_interp = cached_psfs[psf_id_string]
        interp = deepcopy(position_interp)
        image = gs.convolve(interp, webbpsf_interp, num_pix)
        final_array = image.array
        position_images.append(final_array)
        position_lenses.append(position_lens)

        ps, r = power_spectrum_1d(final_array)
        np.save(os.path.join(save_dir, f'im_det_{detector}_{lens.uid}.npy'), final_array)
        np.save(os.path.join(save_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)

    position_titles = [f'{det}, {pos}' for det, pos in zip(detectors, detector_positions)]
    diagnostic_plot.power_spectrum_check(subhalo_images, position_lenses, position_titles, os.path.join(image_save_dir, f'{lens.uid}_02_positions.png'), oversampled=False)

    # BAND-DEPENDENCE
    bands = ['F106', 'F129', 'F158', 'F184']
    band_lens = deepcopy(control_lens)
    band_images, band_lenses = [], []
    for band in bands:
        lens = deepcopy(band_lens)
        band_model = lens.get_array(num_pix=num_pix * oversample, side=side, band=band)
        band_total_flux_cps = band_lens.get_total_flux_cps(band)
        band_interp = InterpolatedImage(Image(band_model, xmin=0, ymin=0), scale=0.11 / oversample, flux=band_total_flux_cps * 146)
        psf_id_string = get_psf_id_string(control_band, 1, (2048, 2048), oversample)
        webbpsf_interp = cached_psfs[psf_id_string]
        image = gs.convolve(band_interp, webbpsf_interp, num_pix)
        final_array = image.array
        band_images.append(final_array)
        band_lenses.append(lens)

        ps, r = power_spectrum_1d(final_array)
        np.save(os.path.join(save_dir, f'im_band_{band}_{lens.uid}.npy'), final_array)
        np.save(os.path.join(save_dir, f'ps_band_{band}_{lens.uid}.npy'), ps)

    diagnostic_plot.power_spectrum_check(band_images, band_lenses, bands, os.path.join(image_save_dir, f'{lens.uid}_03_bands.png'), oversampled=False)

    # save r; should be same for all, since same size (num_pix, num_pix)
    np.save(os.path.join(save_dir, 'r.npy'), r)

    # use Gaussian PSF
    gaussian_lens = deepcopy(control_lens)
    gaussian_model = deepcopy(control_model)
    total_flux_cps = gaussian_lens.get_total_flux_cps(control_band)
    gaussian_interp = InterpolatedImage(Image(gaussian_model, xmin=0, ymin=0), scale=0.11 / oversample, flux=total_flux_cps * 146)
    gaussian_psf_interp = psf.get_gaussian_psf(fwhm=0.087, oversample=oversample, pixel_scale=0.11)
    image = gs.convolve(gaussian_interp, gaussian_psf_interp, num_pix)
    gaussian_final_array = image.array
    ps, r = power_spectrum_1d(gaussian_final_array)
    np.save(os.path.join(save_dir, f'im_gaussian_psf_{lens.uid}.npy'), gaussian_final_array)
    np.save(os.path.join(save_dir, f'ps_gaussian_psf_{lens.uid}.npy'), ps)
    
    # plot final images with different PSFs
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.log10(subhalo_images[2]))
    ax[0].set_title('WebbPSF')
    ax[1].imshow(np.log10(gaussian_final_array))
    ax[1].set_title('Gaussian PSF')
    v = plot_util.get_v([final_array, gaussian_final_array])
    ax[2].imshow(final_array - gaussian_final_array, cmap='bwr', vmin=-v, vmax=v)
    ax[2].set_title('Residual (WebbPSF - Gaussian PSF)')
    plt.savefig(os.path.join(image_save_dir, f'{lens.uid}_04_psf_compare.png'))
    plt.close()

    print(f'Finished lens {lens.uid}.')


if __name__ == '__main__':
    main()
