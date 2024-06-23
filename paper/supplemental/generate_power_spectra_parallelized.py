import multiprocessing
import os
import sys
from copy import deepcopy
from multiprocessing import Pool

import hydra
import numpy as np
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util

    save_dir = os.path.join(config.machine.array_dir, 'power_spectra')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)

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
    num_lenses = 100
    print(f'Collecting {num_lenses} lenses...')
    pickled_lens_list = os.path.join(config.machine.dir_01, '01_hlwas_sim_detectable_lens_list.pkl')
    lens_list = util.unpickle(pickled_lens_list)[:num_lenses]
    print(f'Collected {len(lens_list)} lens(es).')

    print('Generating power spectra...')
    # tuple the parameters
    tuple_list = [(lens, subhalo_params, imaging_params, save_dir) for lens in lens_list]

    # split up the lenses into batches based on core count
    count = len(lens_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= 36
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
    from mejiro.helpers import gs
    from mejiro.utils import util

    # unpack tuple
    (lens, subhalo_params, imaging_params, save_dir) = tuple
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
                                     detector=1, detector_pos=(2048, 2048), suppress_output=False, validate=False,
                                     check_cache=True)
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
                                     detector=detector, detector_pos=detector_pos, suppress_output=False,
                                     validate=False, check_cache=True)
        ps, r = power_spectrum_1d(gs_images[0])
        np.save(os.path.join(save_dir, f'im_det_{detector}_{lens.uid}.npy'), gs_images[0])
        np.save(os.path.join(save_dir, f'ps_det_{detector}_{lens.uid}.npy'), ps)

    print(f'Finished lens {lens.uid}.')


def check_halo_image_alignment(lens, realization):
    sorted_halos = sorted(realization.halos, key=lambda x: x.mass, reverse=True)

    # get image position
    source_x = lens.kwargs_source_dict['F106']['center_x']
    source_y = lens.kwargs_source_dict['F106']['center_y']
    solver = LensEquationSolver(lens.lens_model_class)
    image_x, image_y = solver.image_position_from_source(sourcePos_x=source_x, sourcePos_y=source_y,
                                                         kwargs_lens=lens.kwargs_lens)

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


if __name__ == '__main__':
    main()
