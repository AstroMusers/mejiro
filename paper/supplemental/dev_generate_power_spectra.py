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

@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    repo_dir = config.machine.repo_dir
    data_dir = config.machine.data_dir
    array_dir = os.path.join(data_dir, 'output')

    # enable use of local modules
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util
    from mejiro.helpers import gs


    save_dir = os.path.join(array_dir, 'power_spectra')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)


    # collect lenses
    num_lenses = 100
    print(f'Collecting {num_lenses} lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list.pkl')
    # pickled_lens_list = '/data/bwedig/mejiro/archive/2024-04-22 pipeline/01/01_hlwas_sim_detectable_lens_list.pkl'
    lens_list = util.unpickle(pickled_lens_list)[:num_lenses]
    # pprint(lens_list)
    print('Collected lenses.')


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


    for lens in tqdm(lens_list):
        print(f'Processing lens {lens.uid}...')

        lens._set_classes()
        
        z_lens = round(lens.z_lens, 2)
        z_source = round(lens.z_source, 2)
        log_m_host = np.log10(lens.main_halo_mass)

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

        util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_8.pkl'), cut_8)
        util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_7.pkl'), cut_7)
        util.pickle(os.path.join(save_dir, f'realization_{lens.uid}_cut_6.pkl'), cut_6)

        lens_cut_6 = deepcopy(lens)
        lens_cut_7 = deepcopy(lens)
        lens_cut_8 = deepcopy(lens)

        lens_cut_6.add_subhalos(cut_6, suppress_output=True)
        lens_cut_7.add_subhalos(cut_7, suppress_output=True)
        lens_cut_8.add_subhalos(cut_8, suppress_output=True)

        lenses = [lens, lens_cut_6, lens_cut_7, lens_cut_8]
        titles = [f'no_subhalos_{lens.uid}', f'cut_6_{lens.uid}', f'cut_7_{lens.uid}',
                    f'cut_8_{lens.uid}']
        models = [i.get_array(num_pix=num_pix * oversample, side=side, band='F106') for i in lenses]

        for sl, model, title in zip(lenses, models, titles):
            print(f'Processing model {title}...')
            gs_images, _ = gs.get_images(sl, [model], ['F106'], input_size=num_pix, output_size=num_pix,
                                        grid_oversample=oversample, psf_oversample=oversample,
                                        detector=1, detector_pos=(2048, 2048), suppress_output=False, validate=False, check_cache=True)
            ps, r = power_spectrum_1d(gs_images[0])
            np.save(os.path.join(save_dir, f'im_subs_{title}.npy'), gs_images[0])
            np.save(os.path.join(save_dir, f'ps_subs_{title}.npy'), ps)
        np.save(os.path.join(save_dir, 'r.npy'), r)

        detectors = [4, 1, 9, 17]
        detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

        for detector, detector_pos in zip(detectors, detector_positions):
            print(f'Processing detector {detector}, {detector_pos}...')
            gs_images, _ = gs.get_images(lens_cut_6, [model], ['F106'], input_size=num_pix, output_size=num_pix,
                                        grid_oversample=oversample, psf_oversample=oversample,
                                        detector=detector, detector_pos=detector_pos, suppress_output=False, validate=False, check_cache=True)
            ps, r = power_spectrum_1d(gs_images[0])
            np.save(os.path.join(save_dir, f'im_det_{detector}_{lens.uid}.npy'), gs_images[0])
            np.save(os.path.join(save_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)


if __name__ == '__main__':
    main()
