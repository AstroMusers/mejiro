import os
import sys
from copy import deepcopy

import galsim
import hydra
import numpy as np
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.helpers import gs
    from mejiro.utils import util

    # set directory for all output of this script
    output_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra')
    util.create_directory_if_not_exists(output_dir)
    # util.clear_directory(output_dir)

    debugging = True
    only_ps = True
    detectors = True
    kappa = False

    if not only_ps:
        num_sample_images = 100

        print('Generating flat images...')
        flat_image_dir = os.path.join(output_dir, 'flat_images')
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

        print('Generating Poisson noise...')
        poisson_noise_dir = os.path.join(output_dir, 'poisson_noise')
        util.create_directory_if_not_exists(poisson_noise_dir)
        util.clear_directory(poisson_noise_dir)
        for i in tqdm(range(num_sample_images)):
            poisson_noise_save_path = os.path.join(poisson_noise_dir, f'poisson_noise_{str(i).zfill(4)}.npy')
            generate_poisson_noise(poisson_noise_save_path, mean)
        print('Generated Poisson noise.')

    # set imaging params
    bands = ['F106']  # , 'F129', 'F184'
    oversample = 5
    num_pix = 45
    side = 4.95

    # set subhalo params
    r_tidal = 0.5
    sigma_sub = 0.055
    subhalo_cone = 5
    los_normalization = 0

    # collect lenses
    # num_lenses = 10
    # print(f'Collecting {num_lenses} lenses...')
    # pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list.pkl')
    pickled_lens_list = '/data/scratch/btwedig/mejiro/archive/2024-05-04 pipeline for sample dataset/pipeline/01/01_hlwas_sim_detectable_lens_list.pkl'
    lens_list = util.unpickle(pickled_lens_list)  # [:num_lenses]
    print(f'Collected {len(lens_list)} lens(es).')

    print('Generating power spectra...')
    for lens in tqdm(lens_list):
        if debugging: print(f'Processing lens {lens.uid}...')

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
            continue

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
            if debugging: print(f'    Processing model {title}...')
            gs_images, _ = gs.get_images(sl, [model], ['F106'], input_size=num_pix, output_size=num_pix,
                                         grid_oversample=oversample, psf_oversample=oversample,
                                         detector=1, detector_pos=(2048, 2048), suppress_output=True, validate=False)
            ps, r = power_spectrum_1d(gs_images[0])
            np.save(os.path.join(output_dir, f'im_subs_{title}_{lens.uid}.npy'), gs_images[0])
            np.save(os.path.join(output_dir, f'ps_subs_{title}_{lens.uid}.npy'), ps)
        np.save(os.path.join(output_dir, 'r.npy'), r)

        if kappa:
            for sl, model, title in zip(lenses, models, titles):
                if debugging: print(f'    Processing model {title} kappa...')
                # generate convergence maps
                if sl.realization is None:
                    kappa = sl.get_macrolens_kappa(num_pix, subhalo_cone)
                else:
                    kappa = sl.get_kappa(num_pix, subhalo_cone)
                kappa_power_spectrum, kappa_r = power_spectrum_1d(kappa)
                np.save(os.path.join(output_dir, f'kappa_ps_{title}_{lens.uid}.npy'), kappa_power_spectrum)
                np.save(os.path.join(output_dir, f'kappa_im_{title}_{lens.uid}.npy'), kappa)
            np.save(os.path.join(output_dir, 'kappa_r.npy'), kappa_r)

        # vary detector and detector position for cut_6 lens (default subhalo population)
        if detectors:
            detectors = [4, 1, 9, 17]
            detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

            for detector, detector_pos in zip(detectors, detector_positions):
                if debugging: print(f'    Processing detector {detector}, {detector_pos}...')
                gs_images, _ = gs.get_images(lenses[1], [models[1]], ['F106'], input_size=num_pix, output_size=num_pix,
                                             grid_oversample=oversample, psf_oversample=oversample,
                                             detector=detector, detector_pos=detector_pos, suppress_output=True, validate=False)
                ps, r = power_spectrum_1d(gs_images[0])
                np.save(os.path.join(output_dir, f'im_det_{detector}_{lens.uid}.npy'), gs_images[0])
                np.save(os.path.join(output_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)

        if debugging: print(f'Finished lens {lens.uid}.')

    print('Done.')


def generate_poisson_noise(save_path, mean):
    num_pix = 45
    poisson_noise = np.random.poisson(mean, (num_pix, num_pix))
    np.save(save_path, poisson_noise)


def generate_gaussian_noise(save_path, mean, stdev):
    '''
    ```
    # generate Gaussian noise
    print('Generating Gaussian noise...')
    gaussian_noise_dir = os.path.join(output_dir, 'gaussian_noise')
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
