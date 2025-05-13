import datetime
import galsim
import hydra
import multiprocessing
import numpy as np
import os
import sys
import time
from copy import deepcopy
from multiprocessing import Pool
from scipy.stats import chi2
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    if config.machine.repo_dir not in sys.path:
        sys.path.append(config.machine.repo_dir)
    from mejiro.utils import util
    from mejiro.instruments.hwo import HWO
    from mejiro.lenses.test import SampleStrongLens

    # script configuration options
    debugging = True
    script_config = {
        'mask_quantile': 0.98,
        'image_radius': 0.01,  # arcsec
        'num_lenses': 24,  # None
        'num_grid_points': 50,
        'rng': galsim.UniformDeviate(42)
    }
    subhalo_params = {
        'masses': np.logspace(7, 9, 50),  # 
        'concentration': 10,
        'r_tidal': 0.5,
        'sigma_sub': 0.055,
        'los_normalization': 0.
    }
    imaging_params = {
        'band': 'J',
        'scene_size': 5,  # arcsec
        'oversample': 1,
        'exposure_time': 10000
    }

    # set up directories for script output
    save_dir = os.path.join(config.machine.data_dir, 'output', 'lowest_detectable_subhalo_mass_hwo')
    util.create_directory_if_not_exists(save_dir)
    util.clear_directory(save_dir)
    image_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)

    # collect lenses
    # if debugging:
    #     pipeline_dir = f'{config.machine.pipeline_dir}_dev'
    # else:
    #     pipeline_dir = config.machine.pipeline_dir
    # print(f'Collecting lenses from {pipeline_dir}')
    # lens_list = lens_util.get_detectable_lenses(pipeline_dir, with_subhalos=False, suppress_output=False)
    # og_count = len(lens_list)
    # lens_list = [lens for lens in lens_list if lens.snr > 50]
    # num_lenses = script_config['num_lenses']
    # if num_lenses is not None:
    #     repeats = int(np.ceil(num_lenses / len(lens_list)))
    #     print(f'Repeating lenses {repeats} time(s)')
    #     lens_list *= repeats
    #     lens_list = lens_list[:num_lenses]
    # print(f'Processing {len(lens_list)} lens(es) of {og_count}')
    # lens_list = np.random.choice(lens_list, num_lenses)
    # lens_list = [lens for lens in lens_list if lens.uid != '00000019' and lens.uid != '00000040']
    # lens_list = [lens for lens in lens_list if lens.uid == '00000040']
    # print(f'Processing lens(es): {[lens.uid for lens in lens_list]}')

    # TODO TEMP
    # print(f'Generating subhalos...')
    # for lens in tqdm(lens_list):
    #     realization = lens.generate_cdm_subhalos()
    #     lens.add_subhalos(realization)
    # print(f'Generated subhalos for {len(lens_list)} lens(es).')
    lens_list = [SampleStrongLens()]

    # calculate number of images to save
    # num_permutations = len(positions) * len(subhalo_params['masses']) * script_config['num_positions']
    # idx_to_save = np.random.randint(low=num_permutations, size=len(lens_list))

    hwo = HWO()
    tuple_list = [
        (
            run, lens, hwo, script_config, imaging_params, subhalo_params, save_dir, image_save_dir)
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
    (_, lens, hwo, script_config, imaging_params, subhalo_params, save_dir, image_save_dir) = tuple
    rng = script_config['rng']
    num_grid_points = script_config['num_grid_points']
    mask_quantile = script_config['mask_quantile']
    band = imaging_params['band']
    oversample = imaging_params['oversample']
    scene_size = imaging_params['scene_size']
    exposure_time = imaging_params['exposure_time']
    masses = subhalo_params['masses']
    concentration = subhalo_params['concentration']

    # get image without subhalo
    synth_no_subhalo = SyntheticImage(lens, hwo, band, arcsec=scene_size, oversample=oversample)
    exposure_no_subhalo = Exposure(synth_no_subhalo, exposure_time=exposure_time, rng=rng, detector_effects=True,
                                   sky_background=True, return_noise=True)

    run = 0
    chi2_array = np.empty((num_grid_points, num_grid_points))
    chi2_array_list = []
    for mass in tqdm(masses, leave=True):
        # compute subhalo parameters
        Rs_angle, alpha_Rs = lens.lens_cosmo.nfw_physical2angle(M=mass, c=concentration)

        for i, (halo_x, halo_y) in tqdm(enumerate(util.make_grid(scene_size, num_grid_points)),
                                        total=num_grid_points ** 2, leave=False):
            # build subhalo parameters
            subhalo_type = 'TNFW'
            kwargs_subhalo = {
                'alpha_Rs': alpha_Rs,
                'Rs': Rs_angle,
                'center_x': halo_x,
                'center_y': halo_y,
                'r_trunc': 5 * Rs_angle
            }

            lens_with_subhalo = deepcopy(lens)
            lens_with_subhalo.add_subhalo(subhalo_type, kwargs_subhalo)
            synth = SyntheticImage(lens_with_subhalo, hwo, band=band, arcsec=scene_size, oversample=oversample,
                                   debugging=False)
            exposure = Exposure(synth, exposure_time=exposure_time, rng=rng,
                                poisson_noise=exposure_no_subhalo.poisson_noise,
                                dark_noise=exposure_no_subhalo.dark_noise, read_noise=exposure_no_subhalo.read_noise,
                                suppress_output=True)

            # build mask based on synthetic image with subhalo
            synth_residual = synth.image - synth_no_subhalo.image
            mask_threshold = np.quantile(np.abs(synth_residual), mask_quantile)
            masked_synth_residual = np.ma.masked_where(np.abs(synth_residual) < mask_threshold, synth_residual)
            masked_synth_residual = util.center_crop_image(masked_synth_residual, exposure.exposure.shape)
            mask = np.ma.getmask(masked_synth_residual)

            # mask images
            masked_exposure_no_subhalo = np.ma.masked_array(exposure_no_subhalo.exposure, mask=mask)
            masked_exposure_with_subhalo = np.ma.masked_array(exposure.exposure, mask=mask)

            # calculate chi square between masked images
            chi_square = stats.chi_square(np.ma.compressed(masked_exposure_with_subhalo),
                                          np.ma.compressed(masked_exposure_no_subhalo))

            # populate chi2_array for this mass which chi2s across the grid
            chi2_array[i // num_grid_points, i % num_grid_points] = chi_square

        chi2_array_list.append(chi2_array)
        chi2_array = np.empty((num_grid_points, num_grid_points))

    # initialize chi2 rv
    pixels_unmasked = masked_synth_residual.count()
    dof = pixels_unmasked - 3
    rv = chi2(dof)

    ldsm = np.empty((num_grid_points, num_grid_points))
    for i in range(num_grid_points):
        for j in range(num_grid_points):
            for mass, chi2_array in zip(masses, chi2_array_list):
                chi2_val = chi2_array[i, j]
                pval = rv.sf(chi2_val)
                if pval < 0.001:
                    ldsm[i, j] = mass
                    break

    np.save(os.path.join(save_dir, f'ldsm_hwo_{lens.uid}.npy'), ldsm)


if __name__ == '__main__':
    main()
