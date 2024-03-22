import multiprocessing
import os
import sys
from copy import deepcopy

import galsim
import hydra
import numpy as np
from pyHalo.preset_models import CDM
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.analysis import ft
    from mejiro.utils import util

    # set directory for all output of this script
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra')
    util.create_directory_if_not_exists(save_dir)

    # prepare directory for lenses
    lens_dir = os.path.join(save_dir, 'lenses')
    util.create_directory_if_not_exists(lens_dir)
    util.clear_directory(lens_dir)

    num_lenses = 1000

    # generate flat image
    print('Generating flat images...')
    flat_image_dir = os.path.join(save_dir, 'flat_images')
    util.create_directory_if_not_exists(flat_image_dir)
    util.clear_directory(flat_image_dir)
    flat_images = []
    for i in tqdm(range(num_lenses)):
        flat_image_save_path = os.path.join(flat_image_dir, f'flat_{str(i).zfill(4)}.npy')
        flat_images.append(generate_flat_image(flat_image_save_path))
    print('Generated flat images.')

    # calculate mean, stdev of flat images
    mean = np.mean(flat_images)
    stdev = np.std(flat_images)

    # generate Poisson noise
    print('Generating Poisson noise...')
    poisson_noise_dir = os.path.join(save_dir, 'poisson_noise')
    util.create_directory_if_not_exists(poisson_noise_dir)
    util.clear_directory(poisson_noise_dir)
    for i in tqdm(range(num_lenses)):
        poisson_noise_save_path = os.path.join(poisson_noise_dir, f'poisson_noise_{str(i).zfill(4)}.npy')
        generate_poisson_noise(poisson_noise_save_path, mean)
    print('Generated Poisson noise.')

    # generate Gaussian noise
    print('Generating Gaussian noise...')
    gaussian_noise_dir = os.path.join(save_dir, 'gaussian_noise')
    util.create_directory_if_not_exists(gaussian_noise_dir)
    util.clear_directory(gaussian_noise_dir)
    for i in tqdm(range(num_lenses)):
        gaussian_noise_save_path = os.path.join(gaussian_noise_dir, f'gaussian_noise_{str(i).zfill(4)}.npy')
        generate_gaussian_noise(gaussian_noise_save_path, mean, stdev)
    print('Generated Gaussian noise.')

    # collect lenses
    print(f'Collecting {num_lenses} lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_skypy_output_lens_list')
    lens_list = util.unpickle(pickled_lens_list)[:num_lenses]
    print('Collected lenses.')

    # generate k_list
    print('Generating theta list...')
    num_pix = 45
    side = 4.95
    theta_list = ft.get_theta_list(0.11, side, num_pix)
    np.save(os.path.join(save_dir, 'theta_list.npy'), theta_list)
    print('Generated theta list.')

    print('Generating power spectra...')
    # tuple the parameters
    tuple_list = [(lens, lens_dir) for lens in lens_list]

    # split up the lenses into batches based on core count
    count = len(lens_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    # for batch in tqdm(batches):
    #     pool = Pool(processes=process_count)
    #     pool.map(generate_power_spectra, batch)     

    for tuple in tqdm(tuple_list):
        generate_power_spectra(tuple)

    print('Done.')


def generate_power_spectra(tuple):
    from mejiro.analysis import ft
    from mejiro.helpers import gs, psf

    # unpack tuple
    (lens, lens_dir) = tuple

    # set subhalo params
    r_tidal = 0.25
    sigma_sub = 0.055
    subhalo_cone = 6
    los_normalization = 0

    # set imaging params
    band = 'F184'
    grid_oversample = 3
    num_pix = 45
    side = 4.95

    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)
    # m_host = lens.get_main_halo_mass()
    m_host = lens.lens_mass
    log_m_host = np.log10(m_host)

    cut_6 = CDM(z_lens,
                z_source,
                sigma_sub=sigma_sub,
                log_mlow=6.,
                log_mhigh=10.,
                log_m_host=log_m_host,
                r_tidal=r_tidal,
                cone_opening_angle_arcsec=subhalo_cone,
                LOS_normalization=los_normalization)

    cut_7 = CDM(z_lens,
                z_source,
                sigma_sub=sigma_sub,
                log_mlow=7.,
                log_mhigh=10.,
                log_m_host=log_m_host,
                r_tidal=r_tidal,
                cone_opening_angle_arcsec=subhalo_cone,
                LOS_normalization=los_normalization)

    cut_8 = CDM(z_lens,
                z_source,
                sigma_sub=sigma_sub,
                log_mlow=8.,
                log_mhigh=10.,
                log_m_host=log_m_host,
                r_tidal=r_tidal,
                cone_opening_angle_arcsec=subhalo_cone,
                LOS_normalization=los_normalization)

    lens_cut_6 = deepcopy(lens)
    lens_cut_7 = deepcopy(lens)
    lens_cut_8 = deepcopy(lens)

    lens_cut_6.add_subhalos(cut_6, suppress_output=True)
    lens_cut_7.add_subhalos(cut_7, suppress_output=True)
    lens_cut_8.add_subhalos(cut_8, suppress_output=True)

    lenses = [lens, lens_cut_6, lens_cut_7, lens_cut_8]
    titles = [f'lens_{lens.uid}_no_subhalos', f'lens_{lens.uid}_cut_6', f'lens_{lens.uid}_cut_7',
              f'lens_{lens.uid}_cut_8']

    # generate models
    models = [i.get_array(num_pix=num_pix * grid_oversample, side=side, band=band) for i in lenses]

    # generate images
    detector = gs.get_random_detector(suppress_output=True)
    detector_pos = gs.get_random_detector_pos(num_pix, suppress_output=True)

    for sl, model, title in zip(lenses, models, titles):
        # generate subhalo images and save power spectra
        gs_images, _ = gs.get_images(sl, model, band, input_size=num_pix, output_size=num_pix,
                                     grid_oversample=grid_oversample, psf_oversample=grid_oversample,
                                     detector=detector, detector_pos=detector_pos, suppress_output=True)
        image_power_spectrum = ft.power_spectrum(gs_images[0])
        np.save(os.path.join(lens_dir, f'power_spectrum_{title}_image.npy'), image_power_spectrum)

        # generate convergence maps
        if sl.realization is None:
            kappa = sl.get_macrolens_kappa(num_pix, subhalo_cone)
        else:
            kappa = sl.get_kappa(num_pix, subhalo_cone)
        kappa_power_spectrum = ft.power_spectrum(kappa)
        np.save(os.path.join(lens_dir, f'power_spectrum_{title}_kappa.npy'), kappa_power_spectrum)

    # generate PSF power spectra
    # no PSF
    no_psf_lens = deepcopy(lens)
    no_psf = no_psf_lens.get_array(num_pix=num_pix, side=side, band=band)

    # Gaussian PSF
    gaussian_psf_lens = deepcopy(lens)
    # PSF FWHM for F184; see https://roman.gsfc.nasa.gov/science/WFI_technical.html
    kwargs_psf_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': 0.151}  # TODO get FWHM from roman_params
    gaussian_psf = gaussian_psf_lens.get_array(num_pix=num_pix, side=side, band=band, kwargs_psf=kwargs_psf_gaussian)

    # WebbPSF
    webbpsf_lens = deepcopy(lens)
    webbpsf_kernel = psf.get_psf_kernel(band, detector=1, detector_position=(2048, 2048), oversample=grid_oversample)
    kwargs_webbpsf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': webbpsf_kernel,
        'point_source_supersampling_factor': 5
    }
    webbpsf_psf = webbpsf_lens.get_array(band=band, num_pix=num_pix, kwargs_psf=kwargs_webbpsf, side=side)

    no_psf_power = ft.power_spectrum(no_psf)
    gaussian_psf_power = ft.power_spectrum(gaussian_psf)
    webbpsf_psf_power = ft.power_spectrum(webbpsf_psf)

    np.save(os.path.join(lens_dir, f'power_spectrum_{title}_psf_none.npy'), no_psf_power)
    np.save(os.path.join(lens_dir, f'power_spectrum_{title}_psf_gaussian.npy'), gaussian_psf_power)
    np.save(os.path.join(lens_dir, f'power_spectrum_{title}_psf_webbpsf.npy'), webbpsf_psf_power)


def generate_poisson_noise(save_path, mean):
    num_pix = 45
    poisson_noise = np.random.poisson(mean, (num_pix, num_pix))
    np.save(save_path, poisson_noise)


def generate_gaussian_noise(save_path, mean, stdev):
    num_pix = 45
    gaussian_noise = np.random.normal(mean, stdev, (num_pix, num_pix))
    np.save(save_path, gaussian_noise)


def generate_flat_image(save_path):
    from mejiro.helpers import gs

    band = 'F184'
    num_pix = 45
    ra = 30
    dec = -30
    wcs = gs.get_wcs(ra, dec)
    exposure_time = 146
    detector = 1
    seed = 42

    # calculate sky backgrounds for each band
    bkgs = gs.get_sky_bkgs(wcs, band, detector, exposure_time, num_pix=num_pix)

    # draw interpolated image at the final pixel scale
    im = galsim.ImageF(num_pix, num_pix,
                       scale=0.11)  # NB setting dimensions to "input_size" because we'll crop down to "output_size" at the very end
    im.setOrigin(0, 0)

    # add sky background to convolved image
    final_image = im + bkgs[band]

    # integer number of photons are being detected, so quantize
    final_image.quantize()

    # add all detector effects
    # create galsim rng
    rng = galsim.UniformDeviate(seed=seed)
    galsim.roman.allDetectorEffects(final_image, prev_exposures=(), rng=rng, exptime=exposure_time)

    # make sure there are no negative values from Poisson noise generator
    final_image.replaceNegative()

    # get the array
    final_array = final_image.array

    # divide through by exposure time to get in units of e-/sec/pixel
    final_array /= exposure_time

    # save
    np.save(save_path, final_array)

    return final_array


if __name__ == '__main__':
    main()
