import multiprocessing
import os
import sys
from copy import deepcopy
from lenstronomy.Util.correlation import power_spectrum_1d
import galsim
import hydra
import numpy as np
from pyHalo.preset_models import CDM
from tqdm import tqdm
from multiprocessing import Pool


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.analysis import ft
    from mejiro.utils import util

    # set directory for all output of this script
    lens_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra')
    util.create_directory_if_not_exists(lens_dir)

    # prepare directory for lenses
    lens_dir = os.path.join(lens_dir, 'lenses')
    util.create_directory_if_not_exists(lens_dir)
    util.clear_directory(lens_dir)

    num_sample_images = 100

    # generate flat image
    print('Generating flat images...')
    flat_image_dir = os.path.join(lens_dir, 'flat_images')
    util.create_directory_if_not_exists(flat_image_dir)
    util.clear_directory(flat_image_dir)
    flat_images = []
    for i in tqdm(range(num_sample_images)):
        flat_image_save_path = os.path.join(flat_image_dir, f'flat_{str(i).zfill(4)}.npy')
        flat_images.append(generate_flat_image(flat_image_save_path))
    print('Generated flat images.')

    # calculate mean, stdev of flat images
    mean = np.mean(flat_images)
    stdev = np.std(flat_images)

    # generate Poisson noise
    print('Generating Poisson noise...')
    poisson_noise_dir = os.path.join(lens_dir, 'poisson_noise')
    util.create_directory_if_not_exists(poisson_noise_dir)
    util.clear_directory(poisson_noise_dir)
    for i in tqdm(range(num_sample_images)):
        poisson_noise_save_path = os.path.join(poisson_noise_dir, f'poisson_noise_{str(i).zfill(4)}.npy')
        generate_poisson_noise(poisson_noise_save_path, mean)
    print('Generated Poisson noise.')

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

    # collect lenses
    # num_lenses = 10
    # print(f'Collecting {num_lenses} lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list')  # .pkl
    lens_list = util.unpickle(pickled_lens_list)  # [:num_lenses]
    print(f'Collected {len(lens_list)} lens(es).')

    print('Generating power spectra...')
    # tuple the parameters
    tuple_list = [(lens, lens_dir, subhalo_params, imaging_params) for lens in lens_list]

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
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(generate_power_spectra, batch)     

    print('Done.')


def generate_power_spectra(tuple):
    from mejiro.analysis import ft
    from mejiro.helpers import gs, psf

    # unpack tuple
    (lens, lens_dir, subhalo_params, imaging_params) = tuple
    r_tidal = subhalo_params['r_tidal']
    sigma_sub = subhalo_params['sigma_sub']
    subhalo_cone = subhalo_params['subhalo_cone']
    los_normalization = subhalo_params['los_normalization']
    bands = imaging_params['bands']
    oversample = imaging_params['oversample']
    num_pix = imaging_params['num_pix']
    side = imaging_params['side']

    print(f'Processing lens {lens.uid}...')

    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)
    log_m_host = np.log10(lens.main_halo_mass)

    try:
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
    except:
        print(f'Failed to generate subhalos for lens {lens.uid}.')
        return

    lens_cut_6 = deepcopy(lens)
    lens_cut_7 = deepcopy(lens)
    lens_cut_8 = deepcopy(lens)

    lens_cut_6.add_subhalos(cut_6, suppress_output=True)
    lens_cut_7.add_subhalos(cut_7, suppress_output=True)
    lens_cut_8.add_subhalos(cut_8, suppress_output=True)

    lenses = [lens, lens_cut_6, lens_cut_7, lens_cut_8]
    titles = ['no_subhalos', 'cut_6', 'cut_7', 'cut_8']
    models = [i.get_array(num_pix=num_pix * oversample, side=side, band='F106') for i in lenses]

    for sl, model, title in zip(lenses, models, titles):
        print(f'    Processing model {title}...')
        gs_images, _ = gs.get_images(sl, model, 'F106', input_size=num_pix, output_size=num_pix,
                                    grid_oversample=oversample, psf_oversample=oversample,
                                    detector=1, detector_pos=(2048, 2048), suppress_output=True)
        ps, r = power_spectrum_1d(gs_images[0])
        np.save(os.path.join(lens_dir, f'im_subs_{title}.npy'), gs_images[0])
        np.save(os.path.join(lens_dir, f'ps_subs_{title}.npy'), ps)
    np.save(os.path.join(lens_dir, 'r.npy'), r)

    for sl, model, title in zip(lenses, models, titles):
        print(f'    Processing model {title} kappa...')
    # generate convergence maps
        if sl.realization is None:
            kappa = sl.get_macrolens_kappa(num_pix, subhalo_cone)
        else:
            kappa = sl.get_kappa(num_pix, subhalo_cone)
        kappa_power_spectrum, kappa_r = power_spectrum_1d(kappa)
        np.save(os.path.join(lens_dir, f'kappa_ps_{title}_{lens.uid}.npy'), kappa_power_spectrum)
        np.save(os.path.join(lens_dir, f'kappa_im_{title}_{lens.uid}.npy'), kappa)
    np.save(os.path.join(lens_dir, 'kappa_r.npy'), kappa_r)

    # vary detector and detector position for cut_6 lens (default subhalo population)
    detectors = [4, 1, 9, 17]
    detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

    for detector, detector_pos in zip(detectors, detector_positions):
        print(f'    Processing detector {detector}, {detector_pos}...')
        gs_images, _ = gs.get_images(lenses[1], models[1], 'F106', input_size=num_pix, output_size=num_pix,
                                    grid_oversample=oversample, psf_oversample=oversample,
                                    detector=detector, detector_pos=detector_pos, suppress_output=True)
        ps, r = power_spectrum_1d(gs_images[0])
        np.save(os.path.join(lens_dir, f'im_det_{detector}_{lens.uid}.npy'), gs_images[0])
        np.save(os.path.join(lens_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)

    print(f'Finished lens {lens.uid}.')


def generate_poisson_noise(save_path, mean):
    num_pix = 45
    poisson_noise = np.random.poisson(mean, (num_pix, num_pix))
    np.save(save_path, poisson_noise)


def generate_gaussian_noise(save_path, mean, stdev):
    '''
    ```
    # generate Gaussian noise
    print('Generating Gaussian noise...')
    gaussian_noise_dir = os.path.join(lens_dir, 'gaussian_noise')
    util.create_directory_if_not_exists(gaussian_noise_dir)
    util.clear_directory(gaussian_noise_dir)
    for i in tqdm(range(num_lenses)):
        gaussian_noise_save_path = os.path.join(gaussian_noise_dir, f'gaussian_noise_{str(i).zfill(4)}.npy')
        generate_gaussian_noise(gaussian_noise_save_path, mean, stdev)
    print('Generated Gaussian noise.')
    ```
    '''

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