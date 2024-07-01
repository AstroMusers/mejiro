import multiprocessing
import os
import sys
from copy import deepcopy
from multiprocessing import Pool

import hydra
import matplotlib.pyplot as plt
import numpy as np
from galsim import InterpolatedImage, Image
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util

    # set top directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra_parallelized')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    image_save_dir = os.path.join(save_dir, 'images')
    util.create_directory_if_not_exists(image_save_dir)

    os.environ['WEBBPSF_PATH'] = "/data/bwedig/STScI/webbpsf-data"

    # collect lenses
    print(f'Collecting lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list.pkl')
    lens_list = util.unpickle(pickled_lens_list)
    print(f'Collected {len(lens_list)} lenses.')

    # require >10^8 M_\odot subhalo alignment with image?
    require_alignment = True

    # set subhalo and imaging params
    subhalo_params = {
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'subhalo_cone': 5,
        'los_normalization': 0
    }
    imaging_params = {
        'bands': ['F106'],
        'oversample': 5,
        'num_pix': 45,
        'side': 4.95
    }

    print('Generating power spectra...')
    # tuple the parameters
    tuple_list = [(lens, subhalo_params, imaging_params, require_alignment, save_dir, image_save_dir) for lens in lens_list]

    # split up the lenses into batches based on core count
    count = len(lens_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= 24
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(generate_power_spectra, batch)

    print('Done.')


def generate_power_spectra(tuple):
    from mejiro.helpers import gs, psf
    from mejiro.lenses import lens_util
    from mejiro.plots import diagnostic_plot, plot_util

    # unpack tuple
    (lens, subhalo_params, imaging_params, require_alignment, save_dir, image_save_dir) = tuple
    r_tidal = subhalo_params['r_tidal']
    sigma_sub = subhalo_params['sigma_sub']
    subhalo_cone = subhalo_params['subhalo_cone']
    los_normalization = subhalo_params['los_normalization']
    bands = imaging_params['bands']
    oversample = imaging_params['oversample']
    num_pix = imaging_params['num_pix']
    side = imaging_params['side']

    print(f'Processing lens {lens.uid}...')

    lens._set_classes()

    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)
    log_m_host = np.log10(lens.main_halo_mass)

    if require_alignment:
        cut_8_good = False
        i = 0

        while not cut_8_good:
            cut_8 = CDM(z_lens,
                        z_source,
                        sigma_sub=sigma_sub,
                        log_mlow=8.,
                        log_mhigh=10.,
                        log_m_host=log_m_host,
                        r_tidal=r_tidal,
                        cone_opening_angle_arcsec=subhalo_cone,
                        LOS_normalization=los_normalization)
            cut_8_good = lens_util.check_halo_image_alignment(lens, cut_8, halo_mass=1e8, halo_sort_massive_first=True, return_halo=False)
            i += 1
        print(f'Generated cut_8 population after {i} iterations.')
    else:
        cut_8 = CDM(z_lens,
                    z_source,
                    sigma_sub=sigma_sub,
                    log_mlow=8.,
                    log_mhigh=10.,
                    log_m_host=log_m_host,
                    r_tidal=r_tidal,
                    cone_opening_angle_arcsec=subhalo_cone,
                    LOS_normalization=los_normalization)

    med = CDM(z_lens,
              z_source,
              sigma_sub=sigma_sub,
              log_mlow=7.,
              log_mhigh=8.,
              log_m_host=log_m_host,
              r_tidal=r_tidal,
              cone_opening_angle_arcsec=subhalo_cone,
              LOS_normalization=los_normalization)

    smol = CDM(z_lens,
               z_source,
               sigma_sub=sigma_sub,
               log_mlow=6.,
               log_mhigh=7.,
               log_m_host=log_m_host,
               r_tidal=r_tidal,
               cone_opening_angle_arcsec=subhalo_cone,
               LOS_normalization=los_normalization)

    cut_7 = cut_8.join(med)
    cut_6 = cut_7.join(smol)

    # TODO do I need these?
    # util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_8.pkl'), cut_8)
    # util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_7.pkl'), cut_7)
    # util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_6.pkl'), cut_6)

    lens_cut_6 = deepcopy(lens)
    lens_cut_7 = deepcopy(lens)
    lens_cut_8 = deepcopy(lens)

    lens_cut_6.add_subhalos(cut_6, suppress_output=True)
    lens_cut_7.add_subhalos(cut_7, suppress_output=True)
    lens_cut_8.add_subhalos(cut_8, suppress_output=True)

    lenses = [lens_cut_6, lens_cut_7, lens_cut_8, lens]
    titles = [f'cut_6_{lens.uid}', f'cut_7_{lens.uid}',
                f'cut_8_{lens.uid}', f'no_subhalos_{lens.uid}']
    models = [i.get_array(num_pix=num_pix * oversample, side=side, band=bands[0]) for i in lenses]

    # check that the models were generated correctly
    diagnostic_plot.power_spectrum_check(models, lenses, titles, os.path.join(image_save_dir, f'{lens.uid}_00_models.png'), oversampled=True)

    # calculate sky backgrounds for each band
    wcs_dict = gs.get_wcs(30, -30, date=None)
    bkgs = gs.get_sky_bkgs(wcs_dict, bands, detector=1, exposure_time=146, num_pix=num_pix)

    # generate the PSFs I'll need for each unique band
    psf_kernels = {}
    for band in bands:
        psf_kernels[band] = psf.get_webbpsf_psf(band, detector=1, detector_position=(2048, 2048), oversample=5,
                                                check_cache=True, suppress_output=False)

    # convolve models with the PSFs
    convolved = []
    for model, lens in zip(models, lenses):
        # get interpolated image
        total_flux_cps = lens.get_total_flux_cps(band)
        interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / oversample, flux=total_flux_cps * 146)

        # convolve image with PSF
        psf_kernel = psf_kernels[band]
        image = gs.convolve(interp, psf_kernel, num_pix)

        convolved.append(image)

    diagnostic_plot.power_spectrum_check(convolved, lenses, titles, os.path.join(image_save_dir, f'{lens.uid}_01_convolved.png'), oversampled=False)

    added_bkg = []
    for image, lens in zip(convolved, lenses):
        image += bkgs[band]  # add sky background to convolved image
        image.quantize()  # integer number of photons are being detected, so quantize

        added_bkg.append(image)

    det_fx = []
    for image, lens, title in zip(added_bkg, lenses, titles):
        # NB no detector effects

        # get the array
        final_array = image.array

        # divide through by exposure time to get in units of counts/sec/pixel
        final_array /= 146

        det_fx.append(final_array)

        ps, r = power_spectrum_1d(final_array)
        np.save(os.path.join(save_dir, f'im_subs_{title}.npy'), final_array)
        np.save(os.path.join(save_dir, f'ps_subs_{title}.npy'), ps)

    diagnostic_plot.power_spectrum_check(det_fx, lenses, titles, os.path.join(image_save_dir, f'{lens.uid}_02_bkg_counts_sec.png'), oversampled=False)

    detectors = [4, 1, 9, 17]
    detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

    for detector, detector_pos in zip(detectors, detector_positions):
        # get interpolated image
        total_flux_cps = lens.get_total_flux_cps(band)
        interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / oversample, flux=total_flux_cps * 146)

        # convolve image with PSF
        webbpsf_interp = psf.get_webbpsf_psf(band, detector=detector, detector_position=detector_pos, oversample=5,
                                                check_cache=True, suppress_output=False)
        image = gs.convolve(interp, webbpsf_interp, num_pix)

        bkgs = gs.get_sky_bkgs(wcs_dict, bands, detector=detector, exposure_time=146, num_pix=num_pix)
        image += bkgs[band]  # add sky background to convolved image
        image.quantize()  # integer number of photons are being detected, so quantize

        # NB no detector effects

        # get the array
        final_array = image.array

        # divide through by exposure time to get in units of counts/sec/pixel
        final_array /= 146

        ps, r = power_spectrum_1d(final_array)
        np.save(os.path.join(save_dir, f'im_det_{detector}_{lens.uid}.npy'), final_array)
        np.save(os.path.join(save_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)

    np.save(os.path.join(save_dir, 'r.npy'), r)

    # use Gaussian PSF
    total_flux_cps = lens.get_total_flux_cps(band)
    interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / 5, flux=total_flux_cps * 146)
    gaussian_psf_interp = psf.get_gaussian_psf(fwhm=0.087, oversample=5, pixel_scale=0.11)
    image = gs.convolve(interp, gaussian_psf_interp, 45)
    bkgs = gs.get_sky_bkgs(wcs_dict, bands, detector=detector, exposure_time=146, num_pix=45)
    image += bkgs[band]  # add sky background to convolved image
    image.quantize()  # integer number of photons are being detected, so quantize
    gaussian_final_array = image.array
    gaussian_final_array /= 146
    ps, r = power_spectrum_1d(gaussian_final_array)
    np.save(os.path.join(save_dir, f'im_gaussian_psf_{lens.uid}.npy'), gaussian_final_array)
    np.save(os.path.join(save_dir, f'ps_gaussian_psf_{lens.uid}.npy'), ps)
    
    # plot final images with different PSFs
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.log10(final_array))
    ax[0].set_title('WebbPSF')
    ax[1].imshow(np.log10(gaussian_final_array))
    ax[1].set_title('Gaussian PSF')
    v = plot_util.get_v([final_array, gaussian_final_array])
    ax[2].imshow(final_array - gaussian_final_array, cmap='bwr', vmin=-v, vmax=v)
    ax[2].set_title('Residual (WebbPSF - Gaussian PSF)')
    plt.savefig(os.path.join(image_save_dir, f'{lens.uid}_03_psf_compare.png'))
    plt.close()

    print(f'Finished lens {lens.uid}.')


if __name__ == '__main__':
    main()
