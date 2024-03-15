import os
import sys
from copy import deepcopy

import galsim
import hydra
import numpy as np
from pyHalo.preset_models import CDM


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.lenses.test import SampleStrongLens
    from mejiro.helpers import gs, pyhalo
    from mejiro.utils import util

    # set save path for everything
    save_dir = os.path.join(config.machine.data_dir, 'output', 'power_spectrum_galsim')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    # generate flat image
    flat_image_save_path = os.path.join('flat.npy')
    generate_flat_image(flat_image_save_path)

    # TODO collect num=1000 lenses
    lens_dir = config.machine.dir_01
    uid_list = [str(i).zfill(8) for i in range(1000)]

    lens_list = util.unpickle(pickled_lens_list)
    count = len(lens_list)

    for uid in uid_list:
    # TODO get lens
    # for loop over subhalo populations, PSFs
    # 2. add subhalos
    # 3. generate image
    # 4. generate power spectrum
    # 5. save 
        
    z_lens = round(lens.z_lens, 2)
    z_source = round(lens.z_source, 2)  

    band = 'F184'
    grid_oversample = 3
    num_pix = 45
    side = 4.95

    # randomly generate CDM subhalos
    log_m_host = np.log10(lens.get_main_halo_mass())
    # TODO calculate r_tidal: the core radius of the host halo in units of the host halo scale radius. Subhalos are distributed in 3D with a cored NFW profile with this core radius; by default, it's 0.25
    r_tidal = 0.25
    cdm_realization = CDM(z_lens, z_source, log_m_host=log_m_host, r_tidal=r_tidal,
                          cone_opening_angle_arcsec=subhalo_cone,
                          LOS_normalization=los_normalization)

    # add subhalos
    lens.add_subhalos(cdm_realization)

    lens = SampleStrongLens()

    # generate subhalos
    no_cut = CDM(lens.z_lens,
                 lens.z_source,
                 cone_opening_angle_arcsec=6.,
                 LOS_normalization=0.,
                 log_mlow=6.,
                 log_mhigh=10.)

    cut_7 = CDM(lens.z_lens,
                lens.z_source,
                cone_opening_angle_arcsec=6.,
                LOS_normalization=0.,
                log_mlow=7.,
                log_mhigh=10.)

    cut_8 = CDM(lens.z_lens,
                lens.z_source,
                cone_opening_angle_arcsec=6.,
                LOS_normalization=0.,
                log_mlow=8.,
                log_mhigh=10.)

    util.pickle(os.path.join(save_dir, 'no_cut'), no_cut)
    util.pickle(os.path.join(save_dir, 'cut_7'), cut_7)
    util.pickle(os.path.join(save_dir, 'cut_8'), cut_8)

    no_cut_masses = [halo.mass for halo in no_cut.halos]
    cut_7_masses = [halo.mass for halo in cut_7.halos]
    cut_8_masses = [halo.mass for halo in cut_8.halos]

    np.save(os.path.join(save_dir, 'no_cut_masses'), no_cut_masses)
    np.save(os.path.join(save_dir, 'cut_7_masses'), cut_7_masses)
    np.save(os.path.join(save_dir, 'cut_8_masses'), cut_8_masses)

    no_cut_lens = deepcopy(lens)
    cut_7_lens = deepcopy(lens)
    cut_8_lens = deepcopy(lens)

    no_cut_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(no_cut))
    cut_7_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(cut_7))
    cut_8_lens.add_subhalos(*pyhalo.realization_to_lensing_quantities(cut_8))

    no_cut_model = no_cut_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
    cut_7_model = cut_7_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
    cut_8_model = cut_8_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)

    np.save(os.path.join(save_dir, 'no_cut_model'), no_cut_model)
    np.save(os.path.join(save_dir, 'cut_7_model'), cut_7_model)
    np.save(os.path.join(save_dir, 'cut_8_model'), cut_8_model)

    lenses = [no_cut_lens, cut_7_lens, cut_8_lens]
    models = [no_cut_model, cut_7_model, cut_8_model]
    titles = ['substructure_no_cut', 'substructure_cut_7', 'substructure_cut_8']

    for lens, model, title in zip(lenses, models, titles):
        gs_images, _ = gs.get_images(lens, model, band, input_size=num_pix, output_size=num_pix,
                                     grid_oversample=grid_oversample)
        np.save(os.path.join(save_dir, f'{title}.npy'), gs_images[0])


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
