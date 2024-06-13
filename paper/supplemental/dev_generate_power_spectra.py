#!/usr/bin/env python
# coding: utf-8

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import hydra
from matplotlib import colors
from hydra import initialize, compose
import pickle
from glob import glob
from pprint import pprint
from tqdm import tqdm
from pyHalo.preset_models import CDM
from pyHalo import plotting_routines
from copy import deepcopy
from lenstronomy.Util.correlation import power_spectrum_1d
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from galsim import InterpolatedImage, Image
from copy import deepcopy
import galsim


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    repo_dir = config.machine.repo_dir
    data_dir = config.machine.data_dir
    array_dir = os.path.join(data_dir, 'output')

    # enable use of local modules
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util
    from mejiro.helpers import gs, psf
    from mejiro.lenses import lens_util
    from mejiro.plots import plot_util

    # save_dir = os.path.join(array_dir, 'power_spectra_dev')
    save_dir = os.path.join(array_dir, 'power_spectra')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

    image_save_dir = os.path.join(save_dir, 'images')
    util.create_directory_if_not_exists(image_save_dir)

    os.environ['WEBBPSF_PATH'] = "/data/bwedig/STScI/webbpsf-data"
    
    # collect lenses
    print(f'Collecting lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list.pkl')
    # pickled_lens_list = '/data/bwedig/mejiro/archive/2024-04-22 pipeline/01/01_hlwas_sim_detectable_lens_list.pkl'
    lens_list = util.unpickle(pickled_lens_list)
    # pprint(lens_list)
    print(f'Collected {len(lens_list)} lenses.')

    # require >10^8 M_\odot subhalo alignment with image?
    require_alignment = True

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


    def check_halo_image_alignment(lens, realization):
        sorted_halos = sorted(realization.halos, key=lambda x: x.mass, reverse=True)

        # get image position
        source_x = lens.kwargs_source_dict['F106']['center_x']
        source_y = lens.kwargs_source_dict['F106']['center_y']
        solver = LensEquationSolver(lens.lens_model_class)
        image_x, image_y = solver.image_position_from_source(sourcePos_x=source_x, sourcePos_y=source_y, kwargs_lens=lens.kwargs_lens)

        for halo in sorted_halos:
            if halo.mass < 1e8:
                break
            
            # calculate distances
            for x, y in zip(image_x, image_y):
                dist = np.sqrt(np.power(halo.x - x, 2) + np.power(halo.y - y, 2))

                # check if halo is within 0.1 arcsec of the image
                if dist < 0.1:
                    return True
        
        return False

    
    def plot(array_list, titles, save_path, oversampled):
        if type(array_list[0]) is not np.ndarray:
            array_list = [i.array for i in array_list]

        f, ax = plt.subplots(2, 4, figsize=(12, 6))
        for i, array in enumerate(array_list):
            axis = ax[0][i].imshow(np.log10(array))
            ax[0][i].set_title(titles[i])
            ax[0][i].axis('off')

        cbar = f.colorbar(axis, ax=ax[0])
        cbar.set_label(r'$\log($Counts/sec$)$', rotation=90)

        res_array = [array_list[3] - array_list[i] for i in range(4)]
        v = plot_util.get_v(res_array)
        for i in range(4):
            axis = ax[1][i].imshow(array_list[3] - array_list[i], cmap='bwr', vmin=-v, vmax=v)
            ax[1][i].set_axis_off()

        cbar = f.colorbar(axis, ax=ax[1])
        cbar.set_label('Counts/sec', rotation=90)

        for i, lens in enumerate(lenses):
            realization = lens.realization
            if realization is not None:
                for halo in realization.halos:
                    if halo.mass > 1e8:
                        if oversampled:
                            coords = lens_util.get_coords(45 * 5, delta_pix=0.11 / 5)
                        else:
                            coords = lens_util.get_coords(45, delta_pix=0.11)
                        ax[1][i].scatter(*coords.map_coord2pix(halo.x, halo.y), s=100, facecolors='none', edgecolors='black')

        plt.savefig(save_path)
        plt.close()


    for lens in tqdm(lens_list):
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
                cut_8_good = check_halo_image_alignment(lens, cut_8)
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
        util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_8.pkl'), cut_8)
        util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_7.pkl'), cut_7)
        util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_6.pkl'), cut_6)

        lens_cut_6 = deepcopy(lens)
        lens_cut_7 = deepcopy(lens)
        lens_cut_8 = deepcopy(lens)

        lens_cut_6.add_subhalos(cut_6, suppress_output=True)
        lens_cut_7.add_subhalos(cut_7, suppress_output=True)
        lens_cut_8.add_subhalos(cut_8, suppress_output=True)

        lenses = [lens_cut_6, lens_cut_7, lens_cut_8, lens]
        titles = [f'cut_6_{lens.uid}', f'cut_7_{lens.uid}',
                    f'cut_8_{lens.uid}', f'no_subhalos_{lens.uid}']
        models = [i.get_array(num_pix=num_pix * oversample, side=side, band='F106') for i in lenses]

        plot(models, titles, os.path.join(image_save_dir, f'{lens.uid}_00_models.png'), oversampled=True)

        # create galsim rng
        rng = galsim.UniformDeviate()

        # calculate sky backgrounds for each band
        wcs_dict = gs.get_wcs(30, -30, date=None)
        bkgs = gs.get_sky_bkgs(wcs_dict, bands, detector=1, exposure_time=146, num_pix=45)

        # generate the PSFs I'll need for each unique band
        psf_kernels = {}
        for band in bands:
            psf_kernels[band] = psf.get_webbpsf_psf(band, detector=1, detector_position=(2048, 2048), oversample=5, check_cache=True, suppress_output=False)

        convolved = []
        for model, lens in zip(models, lenses):
            # get interpolated image
            total_flux_cps = lens.get_total_flux_cps(band)
            interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / 5, flux=total_flux_cps * 146)

            # convolve image with PSF
            psf_kernel = psf_kernels[band]
            image = gs.convolve(interp, psf_kernel, 45)

            convolved.append(image)

        plot(convolved, titles, os.path.join(image_save_dir, f'{lens.uid}_01_convolved.png'), oversampled=False)

        added_bkg = []
        for image, lens in zip(convolved, lenses):
            image += bkgs[band]  # add sky background to convolved image
            image.quantize()  # integer number of photons are being detected, so quantize

            added_bkg.append(image)

        plot(added_bkg, titles, os.path.join(image_save_dir, f'{lens.uid}_02_added_bkg.png'), oversampled=False)

        det_fx = []
        for image, lens, title in zip(added_bkg, lenses, titles):
            # add all detector effects
            # galsim.roman.allDetectorEffects(image, prev_exposures=(), rng=rng, exptime=146)

            # get the array
            final_array = image.array

            # divide through by exposure time to get in units of counts/sec/pixel
            final_array /= 146

            det_fx.append(final_array)

            ps, r = power_spectrum_1d(final_array)
            np.save(os.path.join(save_dir, f'im_subs_{title}.npy'), final_array)
            np.save(os.path.join(save_dir, f'ps_subs_{title}.npy'), ps)

        plot(det_fx, titles, os.path.join(image_save_dir, f'{lens.uid}_03_counts_sec.png'), oversampled=False)

        # for sl, model, title in zip(lenses, models, titles):
        #     print(f'Processing model {title}...')
        #     gs_images, _ = gs.get_images(sl, [model], ['F106'], input_size=num_pix, output_size=num_pix,
        #                                 grid_oversample=oversample, psf_oversample=oversample,
        #                                 detector=1, detector_pos=(2048, 2048), suppress_output=False, validate=False, check_cache=True)
        #     ps, r = power_spectrum_1d(gs_images[0])
        #     np.save(os.path.join(save_dir, f'im_subs_{title}.npy'), gs_images[0])
        #     np.save(os.path.join(save_dir, f'ps_subs_{title}.npy'), ps)
        # np.save(os.path.join(save_dir, 'r.npy'), r)

        detectors = [4, 1, 9, 17]
        detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

        for detector, detector_pos in zip(detectors, detector_positions):
        #     print(f'Processing detector {detector}, {detector_pos}...')
        #     gs_images, _ = gs.get_images(lens_cut_6, [model], ['F106'], input_size=num_pix, output_size=num_pix,
        #                                 grid_oversample=oversample, psf_oversample=oversample,
        #                                 detector=detector, detector_pos=detector_pos, suppress_output=False, validate=False, check_cache=True)

            # get interpolated image
            total_flux_cps = lens.get_total_flux_cps(band)
            interp = InterpolatedImage(Image(model, xmin=0, ymin=0), scale=0.11 / 5, flux=total_flux_cps * 146)

            # convolve image with PSF
            psf_kernel = psf.get_webbpsf_psf(band, detector=detector, detector_position=detector_pos, oversample=5, check_cache=True, suppress_output=False)
            image = gs.convolve(interp, psf_kernel, 45)

            bkgs = gs.get_sky_bkgs(wcs_dict, bands, detector=detector, exposure_time=146, num_pix=45)
            image += bkgs[band]  # add sky background to convolved image
            image.quantize()  # integer number of photons are being detected, so quantize

            # add all detector effects
            # galsim.roman.allDetectorEffects(image, prev_exposures=(), rng=rng, exptime=146)

            # get the array
            final_array = image.array

            # divide through by exposure time to get in units of counts/sec/pixel
            final_array /= 146

            ps, r = power_spectrum_1d(final_array)
            np.save(os.path.join(save_dir, f'im_det_{detector}_{lens.uid}.npy'), final_array)
            np.save(os.path.join(save_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)

        np.save(os.path.join(save_dir, 'r.npy'), r)


if __name__ == '__main__':
    main()
