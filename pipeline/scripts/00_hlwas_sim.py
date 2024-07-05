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
    debugging = True

    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    # set up output directory
    output_dir = config.machine.dir_00
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if debugging: print(f'Set up output directory {output_dir}')

    # load Roman WFI filters
    configure_roman_filters()
    roman_filters = filter_names()
    roman_filters.sort()
    # roman_filters = sorted(glob(os.path.join(repo_dir, 'mejiro', 'data', 'avg_filter_responses', 'Roman-*.ecsv')))
    _ = speclite.filters.load_filters(*roman_filters[:8])
    if debugging:
        print('Configured Roman filters. Loaded:')
        pprint(roman_filters[:8])

    # tuple the parameters
    survey_params = util.hydra_to_dict(config.survey)
    pipeline_params = util.hydra_to_dict(config.pipeline)
    runs = pipeline_params['survey_sim_runs']
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

    # prepare a directory for this particular run
    lens_output_dir = os.path.join(output_dir, f'run_{str(run).zfill(2)}')
    util.create_directory_if_not_exists(lens_output_dir)

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

    # draw the total lens population
    if debugging: print('Identifying lenses...')
    kwargs_lens_total_cut = {
        'min_image_separation': 0,
        'max_image_separation': 10,
        'mag_arc_limit': None
    }
    total_lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_total_cut)
    if debugging: print(f'Number of total lenses: {len(total_lens_population)}')

    # compute SNRs and save
    if debugging: print(f'Computing SNRs for {len(total_lens_population)} lenses')
    snr_list = []
    for candidate in tqdm(total_lens_population, disable=not debugging):
        snr, _ = survey_sim.get_snr(gglens=candidate, 
                                    band=survey_params['snr_band'],
                                    mask_mult=survey_params['snr_mask_multiplier'],
                                    zodi_mult=survey_params['zodi_multiplier'])
        snr_list.append(snr)
    np.save(os.path.join(output_dir, f'snr_list_{str(run).zfill(2)}.npy'), snr_list)

    # save other params to CSV
    total_pop_csv = os.path.join(output_dir, f'total_pop_{str(run).zfill(2)}.csv')
    if debugging: print(f'Writing total population to {total_pop_csv}')
    survey_sim.write_lens_pop_to_csv(total_pop_csv, total_lens_population, bands, suppress_output=not debugging)

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
    limit = None
    detectable_gglenses, snr_list = [], []
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
        snr, _ = survey_sim.get_snr(gglens=candidate, 
                                    band=survey_params['snr_band'],
                                    mask_mult=survey_params['snr_mask_multiplier'],
                                    zodi_mult=survey_params['zodi_multiplier'])

        if snr < survey_params['snr_threshold']:
            snr_list.append(snr)
            filter_2 += 1
            if filter_2 <= num_samples:
                filtered_sample['filter_2'].append(candidate)
            continue

        # if both criteria satisfied, consider detectable
        detectable_gglenses.append(candidate)

        # if I've imposed a limit above this loop, exit the loop
        if limit is not None and len(detectable_gglenses) == limit:
            break

    if debugging: print(f'Run {str(run).zfill(2)}: {len(detectable_gglenses)} detectable lens(es)')

    # save information about which lenses got filtered out
    filtered_sample['num_filter_1'] = filter_1
    filtered_sample['num_filter_2'] = filter_2
    util.pickle(os.path.join(output_dir, f'filtered_sample_{str(run).zfill(2)}.pkl'), filtered_sample)

    # if len(detectable_gglenses) > 0:
    #     print(filtered_sample['num_filter_1']) 
    #     print(filtered_sample['num_filter_2'])

    if debugging: print('Retrieving lenstronomy parameters...')
    dict_list = []
    for gglens, snr in tqdm(zip(detectable_gglenses, snr_list), disable=not debugging, total=len(detectable_gglenses)):

        # get lens params from gglens object
        kwargs_model, kwargs_params = gglens.lenstronomy_kwargs(band='F106')

        # build dicts for lens and source magnitudes
        lens_mags, source_mags = {}, {}
        for band in bands:
            lens_mags[band] = gglens.deflector_magnitude(band)
            source_mags[band] = gglens.extended_source_magnitude(band, lensed=True)

        z_lens, z_source = gglens.deflector_redshift, gglens.source_redshift
        kwargs_lens = kwargs_params['kwargs_lens']

        # add additional necessary key/value pairs to kwargs_model
        kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
        kwargs_model['source_redshift_list'] = [z_source]
        kwargs_model['cosmo'] = cosmo
        kwargs_model['z_source'] = z_source
        kwargs_model['z_source_convention'] = 5.

        # create dict to pickle
        gglens_dict = {
            'kwargs_model': kwargs_model,
            'kwargs_params': kwargs_params,
            'lens_mags': lens_mags,
            'source_mags': source_mags,
            'deflector_stellar_mass': gglens.deflector_stellar_mass(),
            'deflector_velocity_dispersion': gglens.deflector_velocity_dispersion(),
            'snr': snr
        }

        dict_list.append(gglens_dict)

    if debugging: print('Pickling lenses...')
    for i, each in tqdm(enumerate(dict_list), disable=not debugging):
        save_path = os.path.join(lens_output_dir, f'detectable_lens_{str(run).zfill(2)}_{str(i).zfill(5)}.pkl')
        util.pickle(save_path, each)

    detectable_pop_csv = os.path.join(output_dir, f'detectable_pop_{str(run).zfill(2)}.csv')
    survey_sim.write_lens_pop_to_csv(detectable_pop_csv, detectable_gglenses, bands)

    detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{str(run).zfill(2)}.pkl')
    if debugging: print(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
    util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)


if __name__ == '__main__':
    main()
