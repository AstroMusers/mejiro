import datetime
import multiprocessing
import os
import sys
import time
from glob import glob
from multiprocessing import Pool
from pprint import pprint

import hydra
import numpy as np
import speclite
from astropy.cosmology import default_cosmology
from astropy.units import Quantity
from slsim.lens_pop import LensPop
from slsim.Observations.roman_speclite import configure_roman_filters
from slsim.Observations.roman_speclite import filter_names
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # debugging mode will print statements to console
    debugging = False

    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # load Roman WFI filters
    configure_roman_filters()
    roman_filters = filter_names()
    roman_filters.sort()
    # roman_filters = sorted(glob(os.path.join(repo_dir, 'mejiro', 'data', 'avg_filter_responses', 'Roman-*.ecsv')))
    _ = speclite.filters.load_filters(*roman_filters[:8])
    if debugging:
        print('Configured Roman filters. Loaded:')
        pprint(roman_filters[:8])

    # kwargs_deflector_cut = {
    #     'band': survey_params['deflector_cut_band'],
    #     'band_max': survey_params['deflector_cut_band_max'],
    #     'z_min': survey_params['deflector_z_min'],
    #     'z_max': survey_params['deflector_z_max']
    # }
    # kwargs_source_cut = {
    #     'band': survey_params['source_cut_band'],
    #     'band_max': survey_params['source_cut_band_max'],
    #     'z_min': survey_params['source_z_min'],
    #     'z_max': survey_params['source_z_max']
    # }

    # tuple the parameters
    survey_params = util.hydra_to_dict(config.survey)
    pipeline_params = util.hydra_to_dict(config.pipeline)
    runs = pipeline_params['survey_sim_runs']

    deflector_source_pairs = [(21, 22), (21, 23), (21, 24), (23, 24), (23, 25), (23, 26), (24, 25), (24, 26), (24, 27), (25, 26), (25, 27), (25, 28), (26, 27), (26, 28), (26, 29), (27, 28), (27, 29), (27, 30)]

    for deflector_cut, source_cut in tqdm(deflector_source_pairs):
            survey_params['deflector_cut_band_max'] = deflector_cut
            survey_params['source_cut_band_max'] = source_cut

            output_dir = os.path.join(config.machine.data_dir, 'survey_params', f'deflector_{deflector_cut}_source_{source_cut}')
            util.create_directory_if_not_exists(output_dir)
            util.clear_directory(output_dir)
            if debugging: print(f'Set up output directory {output_dir}')

            tuple_list = [(run, survey_params, pipeline_params, output_dir, debugging) for run in range(runs)]

            # split up the lenses into batches based on core count
            cpu_count = multiprocessing.cpu_count()
            process_count = cpu_count - config.machine.headroom_cores
            count = runs
            if count < process_count:
                process_count = count
            print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

            # batch
            generator = util.batch_list(tuple_list, process_count)
            batches = list(generator)

            # process the batches
            for batch in tqdm(batches):
                pool = Pool(processes=process_count)
                pool.map(run_slsim, batch)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def run_slsim(tuple):
    # a legacy function but prevents duplicate runs
    np.random.seed()

    import mejiro
    from mejiro.helpers import survey_sim
    from mejiro.utils import util

    # unpack tuple
    run, survey_params, pipeline_params, output_dir, debugging = tuple

    # load SkyPy config file
    module_path = os.path.dirname(mejiro.__file__)
    skypy_config = os.path.join(module_path, 'data', 'roman_hlwas.yml')
    if debugging: print(f'Loaded SkyPy configuration file {skypy_config}')

    # set HLWAS parameters
    config_file = util.load_skypy_config(skypy_config)  # read skypy config file to get survey area
    survey_area = float(config_file['fsky'][:-5])
    sky_area = Quantity(value=survey_area, unit='deg2')
    from astropy.cosmology import FlatLambdaCDM
    # cosmo = default_cosmology.get() TODO UPDATE
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    bands = pipeline_params['bands']
    if debugging: print(f'Surveying {sky_area.value} deg2 with bands {bands}')

    # define cuts on the intrinsic deflector and source populations (in addition to the skypy config file)
    kwargs_deflector_cut = {
        'band': survey_params['deflector_cut_band'],
        'band_max': survey_params['deflector_cut_band_max'],
        'z_min': survey_params['deflector_z_min'],
        'z_max': survey_params['deflector_z_max']
    }
    kwargs_source_cut = {
        'band': survey_params['source_cut_band'],
        'band_max': survey_params['source_cut_band_max'],
        'z_min': survey_params['source_z_min'],
        'z_max': survey_params['source_z_max']
    }

    # create the lens population
    if debugging: print('Defining galaxy population...')
    lens_pop = LensPop(deflector_type="all-galaxies",
                       source_type="galaxies",
                       kwargs_deflector_cut=kwargs_deflector_cut,
                       kwargs_source_cut=kwargs_source_cut,
                       kwargs_mass2light=None,
                       skypy_config=skypy_config,
                       sky_area=sky_area,
                       cosmo=cosmo)
    if debugging: print('Defined galaxy population')

    # draw initial detectable lens population
    if debugging: print('Identifying detectable lenses...')
    kwargs_lens_detectable_cut = {
        'min_image_separation': survey_params['min_image_separation'],
        'max_image_separation': survey_params['max_image_separation'],
        'mag_arc_limit': {survey_params['mag_arc_limit_band']: survey_params['mag_arc_limit']}
    }
    lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_detectable_cut)
    if debugging: print(f'Number of detectable lenses from first set of criteria: {len(lens_population)}')

    # set up dict to capture some information about which candidates got filtered out
    filtered_sample = {}
    filtered_sample['total'] = len(lens_population)
    num_samples = 16
    filter_1, filter_2 = 0, 0
    filtered_sample['filter_1'] = []
    filtered_sample['filter_2'] = []

    # apply additional detectability criteria
    detectable_gglenses, snr_list, complete_snr_list = [], [], []
    for candidate in tqdm(lens_population, disable=not debugging):
        # 1. Einstein radius and Sersic radius
        _, kwargs_params = candidate.lenstronomy_kwargs(band=survey_params['large_lens_band'])
        lens_mag = candidate.deflector_magnitude(band=survey_params['large_lens_band'])

        if kwargs_params['kwargs_lens'][0]['theta_E'] < kwargs_params['kwargs_lens_light'][0][
            'R_sersic'] and lens_mag < survey_params['large_lens_mag_max']:
            filter_1 += 1
            if filter_1 <= num_samples:
                filtered_sample['filter_1'].append(candidate)
            continue

        # 2. SNR
        snr, _ = survey_sim.get_snr(candidate, survey_params['snr_band'],
                                    mask_mult=survey_params['snr_mask_multiplier'])
        complete_snr_list.append(snr)

        if snr < survey_params['snr_threshold']:
            snr_list.append(snr)
            filter_2 += 1
            if filter_2 <= num_samples:
                filtered_sample['filter_2'].append(candidate)
            continue

        # if both criteria satisfied, consider detectable
        detectable_gglenses.append(candidate)

    if debugging: print(f'Run {str(run).zfill(2)}: {len(detectable_gglenses)} detectable lens(es)')

    filtered_sample['num_filter_1'] = filter_1
    filtered_sample['num_filter_2'] = filter_2
    util.pickle(os.path.join(output_dir, f'filtered_sample_{str(run).zfill(2)}.pkl'), filtered_sample)

    complete_snr_list_path = os.path.join(output_dir, f'complete_snr_list_{str(run).zfill(2)}.pkl')
    util.pickle(complete_snr_list_path, complete_snr_list)

    detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{str(run).zfill(2)}.pkl')
    if debugging: print(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
    util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)


if __name__ == '__main__':
    main()
