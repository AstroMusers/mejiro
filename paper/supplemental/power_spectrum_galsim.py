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
    from mejiro.helpers import gs
    from mejiro.utils import util

    # set save path for everything
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectra')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    # generate flat image
    print('Generating flat image...')
    flat_image_save_path = os.path.join(save_dir, 'flat.npy')
    generate_flat_image(flat_image_save_path)
    print('Generated flat image.')

    # collect lenses
    num_lenses = 100
    print(f'Collecting {num_lenses} lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_skypy_output_lens_list')
    lens_list = util.unpickle(pickled_lens_list)[:num_lenses]
    print('Collected lenses.')

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

    # generate k_list
    print('Generating k list...')
    k_list = ft.get_k_list(0.11, side, num_pix)
    np.save(os.path.join(save_dir, 'k_list.npy'), k_list)
    print('Generated k list.')

    # generate subhalo power spectra
    print('Generating subhalo power spectra...')
    for lens in tqdm(lens_list):
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
        titles = [f'lens_{lens.uid}_no_subhalos', f'lens_{lens.uid}_cut_6', f'lens_{lens.uid}_cut_7', f'lens_{lens.uid}_cut_8']

        # generate models
        models = [lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band) for lens in lenses]

        # generate images
        detector = gs.get_random_detector(suppress_output=True)
        detector_pos = gs.get_random_detector_pos(num_pix, suppress_output=True)

        for lens, model, title in zip(lenses, models, titles):
            gs_images, _ = gs.get_images(lens, model, band, input_size=num_pix, output_size=num_pix,
                                            grid_oversample=grid_oversample, psf_oversample=grid_oversample, 
                                            detector=detector, detector_pos=detector_pos, suppress_output=True)

            # generate power spectrum
            power_spectrum = ft.power_spectrum(gs_images[0])
            np.save(os.path.join(save_dir, f'power_spectrum_{lens.uid}_{title}.npy'), power_spectrum)
    print('Generated subhalo power spectra.')


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

    # divide through by exposure time to get in units of counts/sec/pixel
    final_array /= exposure_time

    # save
    np.save(save_path, final_array)


if __name__ == '__main__':
    main()
