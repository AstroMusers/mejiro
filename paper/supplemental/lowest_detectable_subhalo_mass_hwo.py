import datetime
import multiprocessing
import os
import sys
import time
from copy import deepcopy
from multiprocessing import Pool

import galsim
import hydra
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.helpers import survey_sim
    from mejiro.utils import util
    from mejiro.instruments.roman import Roman

    # script configuration options
    debugging = True
    script_config = {
        'image_radius': 0.1,  # arcsec
        'num_lenses': 10,
        'num_positions': 100,
        'rng': galsim.UniformDeviate(42)
    }
    subhalo_params = {
        'masses': np.linspace(1e6, 1e10, 10),
        'concentration': 6,
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'band': 'F106',
        'scene_size': 5,  # arcsec
        'oversample': 5,
        'exposure_time': 146
    }
    positions = [
        # ('01', (2048, 2048)),
        # ('02', (2048, 2048)),
        # ('03', (2048, 2048)),
        ('04', (2048, 2048)),
        # ('05', (2048, 2048)),
        # ('06', (2048, 2048)),
        # ('07', (2048, 2048)),
        # ('08', (2048, 2048)),
        # ('09', (2048, 2048)),
        # ('10', (2048, 2048)),
        ('11', (2048, 2048)),
        # ('12', (2048, 2048)),
        # ('13', (2048, 2048)),
        # ('14', (2048, 2048)),
        # ('15', (2048, 2048)),
        ('16', (2048, 2048)),
        # ('17', (2048, 2048)),
        # ('18', (2048, 2048))
    ]

    # set up directories for script output
    save_dir = os.path.join(config.machine.data_dir, 'output', 'lowest_detectable_subhalo_mass')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    # debugging? set pipeline dir accordingly
    pipeline_params = util.hydra_to_dict(config.pipeline)
    # TODO set debugging here based on yaml file? or manually above?
    if debugging:
        pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    else:
        pipeline_dir = config.machine.pipeline_dir

    # collect lenses
    print(f'Collecting lenses...')
    lens_list = survey_sim.collect_all_detectable_lenses(os.path.join(pipeline_dir, '01'), suppress_output=False)
    num_lenses = script_config['num_lenses']
    print(f'Collected {len(lens_list)} lens(es) and processing {num_lenses}.')
    lens_list = lens_list[:num_lenses]

    print(f'Generating subhalos...')
    for lens in tqdm(lens_list):
        realization = lens.generate_cdm_subhalos()
        lens.add_subhalos(realization)
    print(f'Generated subhalos for {len(lens_list)} lens(es).')

    roman = Roman()
    tuple_list = [
        (
        run, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir, debugging)
        for
        run, lens in enumerate(lens_list)]

    # split up the lenses into batches based on core count
    count = len(lens_list)
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(run, batch)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

    execution_time_per_lens = str(datetime.timedelta(seconds=round((stop - start) / len(lens_list))))
    print(f'Execution time per lens: {execution_time_per_lens}')


def run(tuple):
    from mejiro.analysis import stats
    from mejiro.utils import util
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.exposure import Exposure

    # unpack tuple
    (run, lens, roman, script_config, imaging_params, subhalo_params, positions, save_dir, image_save_dir,
     debugging) = tuple
    image_radius = script_config['image_radius']
    num_positions = script_config['num_positions']
    rng = script_config['rng']
    band = imaging_params['band']
    oversample = imaging_params['oversample']
    scene_size = imaging_params['scene_size']
    exposure_time = imaging_params['exposure_time']
    masses = subhalo_params['masses']
    concentration = subhalo_params['concentration']

    # solve for image positions
    image_x, image_y = lens.get_image_positions(pixel_coordinates=False)
    num_images = len(image_x)

    # TODO a diagnostic plot that shows image positions and lens configuration

    results = {}
    for sca, sca_position in positions:
        sca = int(sca)
        position_key = f'{sca}_{sca_position[0]}_{sca_position[1]}'
        results[position_key] = {}
        for m200 in masses:
            mass_key = f'{m200:.2f}'
            results[position_key][mass_key] = []

            Rs_angle, alpha_Rs = lens.lens_cosmo.nfw_physical2angle(M=m200, c=concentration)
            chi_square_list = []
            for i in range(num_positions):
                execution_key = f'{position_key}_{mass_key}_{str(i).zfill(8)}'

                rand_idx = np.random.randint(0, num_images)  # pick a random image position
                halo_x = image_x[rand_idx]
                halo_y = image_y[rand_idx]

                # get a random point within a small disk around the image
                r = image_radius * np.sqrt(np.random.random_sample())
                theta = 2 * np.pi * np.random.random_sample()
                delta_x, delta_y = util.polar_to_cartesian(r, theta)

                # set subhalo position
                center_x = halo_x + delta_x
                center_y = halo_y + delta_y

                # build subhalo parameters
                subhalo_type = 'TNFW'
                kwargs_subhalo = {
                    'alpha_Rs': alpha_Rs,
                    'Rs': Rs_angle,
                    'center_x': center_x,
                    'center_y': center_y,
                    'r_trunc': 5 * Rs_angle
                }

                # get image with no subhalo
                lens_no_subhalo = deepcopy(lens)
                synth_no_subhalo = SyntheticImage(lens_no_subhalo, roman, band=band, arcsec=scene_size,
                                                  oversample=oversample, sca=sca, sca_position=sca_position)
                exposure_no_subhalo = Exposure(synth_no_subhalo, exposure_time=exposure_time, rng=rng, sca=sca,
                                               sca_position=sca_position, return_noise=True)

                # get noise
                poisson_noise = exposure_no_subhalo.poisson_noise
                dark_noise = exposure_no_subhalo.dark_noise
                read_noise = exposure_no_subhalo.read_noise

                # get image with subhalo
                lens_with_subhalo = deepcopy(lens)
                lens_with_subhalo.add_subhalo(subhalo_type, kwargs_subhalo)
                synth = SyntheticImage(lens_with_subhalo, roman, band=band, arcsec=scene_size, oversample=oversample,
                                       sca=sca, sca_position=sca_position)
                exposure = Exposure(synth, exposure_time=exposure_time, rng=rng, sca=sca, sca_position=sca_position,
                                    poisson_noise=poisson_noise, dark_noise=dark_noise, read_noise=read_noise)

                # calculate chi square
                chi_square = stats.chi_square(observed=exposure.exposure, expected=exposure_no_subhalo.exposure)
                chi_square_list.append(chi_square)

                # save image
                if i % 100 == 0:
                    residual = exposure.exposure - exposure_no_subhalo.exposure
                    vmax = np.max(np.abs(residual))
                    plt.imshow(residual, cmap='bwr', vmin=-vmax, vmax=vmax)
                    plt.colorbar()
                    plt.title(r'$\chi^2=$ ' + f'{chi_square:.2f}')
                    plt.savefig(os.path.join(image_save_dir, f'{execution_key}.png'))
                    plt.close()

            results[position_key][mass_key] = chi_square_list

    util.pickle(os.path.join(save_dir, f'results_{lens.uid}.pkl'), results)


if __name__ == '__main__':
    main()
