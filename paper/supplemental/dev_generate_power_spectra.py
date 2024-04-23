#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
import hydra


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.helpers import gs
    from mejiro.utils import util


# Choose where to save

# In[ ]:


    save_dir = os.path.join(config.machine.array_dir, 'power_spectra')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)


    # In[ ]:


    # collect lenses
    num_lenses = 32
    print(f'Collecting {num_lenses} lenses...')
    # pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list')
    pickled_lens_list = '/data/bwedig/mejiro/archive/2024-04-22 pipeline/01/01_hlwas_sim_detectable_lens_list'
    lens_list = util.unpickle(pickled_lens_list)[:num_lenses]
    pprint(lens_list)
    print('Collected lenses.')


    # In[ ]:


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


    # In[ ]:


    for lens in lens_list:
        print(f'Processing lens {lens.uid}...')
        z_lens = round(lens.z_lens, 2)
        z_source = round(lens.z_source, 2)
        log_m_host = np.log10(lens.main_halo_mass)

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
        models = [i.get_array(num_pix=num_pix * oversample, side=side, band='F106') for i in lenses]

        for sl, model, title in zip(lenses, models, titles):
            print(f'Processing model {title}...')
            gs_images, _ = gs.get_images(sl, model, 'F106', input_size=num_pix, output_size=num_pix,
                                        grid_oversample=oversample, psf_oversample=oversample,
                                        detector=1, detector_pos=(2048, 2048), suppress_output=True)
            ps, r = power_spectrum_1d(gs_images[0])
            np.save(os.path.join(save_dir, f'im_subs_{title}.npy'), gs_images[0])
            np.save(os.path.join(save_dir, f'ps_subs_{title}.npy'), ps)
        np.save(os.path.join(save_dir, 'r.npy'), r)

        # for sl, model, title in zip(lenses, models, titles):
        #     print(f'    Processing model {title} kappa...')
        # # generate convergence maps
        #     if sl.realization is None:
        #         kappa = sl.get_macrolens_kappa(num_pix, subhalo_cone)
        #     else:
        #         kappa = sl.get_kappa(num_pix, subhalo_cone)
        #     kappa_power_spectrum, kappa_r = power_spectrum_1d(kappa)
        #     np.save(os.path.join(save_dir, f'kappa_ps_{title}.npy'), kappa_power_spectrum)
        #     np.save(os.path.join(save_dir, f'kappa_im_{title}.npy'), kappa)
        # np.save(os.path.join(save_dir, 'kappa_r.npy'), kappa_r)


    # In[ ]:


    for lens in lens_list:
        print(f'Processing lens {lens.uid}...')
        z_lens = round(lens.z_lens, 2)
        z_source = round(lens.z_source, 2)
        log_m_host = np.log10(lens.main_halo_mass)

        cut_6 = CDM(z_lens,
                    z_source,
                    sigma_sub=sigma_sub,
                    log_mlow=6.,
                    log_mhigh=10.,
                    log_m_host=log_m_host,
                    r_tidal=r_tidal,
                    cone_opening_angle_arcsec=subhalo_cone,
                    LOS_normalization=los_normalization)
        
        lens_cut_6.add_subhalos(cut_6, suppress_output=True)
        
        detectors = [4, 1, 9, 17]
        detector_positions = [(4, 4092), (2048, 2048), (4, 4), (4092, 4092)]

        for detector, detector_pos in zip(detectors, detector_positions):
            print(f'Processing detector {detector}, {detector_pos}...')
            gs_images, _ = gs.get_images(sl, model, 'F106', input_size=num_pix, output_size=num_pix,
                                        grid_oversample=oversample, psf_oversample=oversample,
                                        detector=detector, detector_pos=detector_pos, suppress_output=True)
            ps, r = power_spectrum_1d(gs_images[0])
            np.save(os.path.join(save_dir, f'im_det_{detector}.npy'), gs_images[0])
            np.save(os.path.join(save_dir, f'ps_det_{detector}.npy'), ps)


if __name__ == '__main__':
    main()
