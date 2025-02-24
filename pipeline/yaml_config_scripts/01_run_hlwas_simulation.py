import json
import multiprocessing
import os
import sys
import time
import yaml
from glob import glob
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import speclite
from astropy.cosmology import default_cosmology
from astropy.units import Quantity
from slsim.lens_pop import LensPop
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors
from tqdm import tqdm


SCRIPT_NAME = '01'


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util
    from mejiro.lenses import lens_util

    # retrieve configuration parameters
    dev = config['pipeline']['dev']
    verbose = config['pipeline']['verbose']
    data_dir = config['data_dir']
    runs = config['pipeline']['survey']['runs']
    scas = config['pipeline']['survey']['scas']
    area = config['pipeline']['survey']['area']

    # set nice level
    os.nice(config['pipeline']['nice'])

    # set up top directory for all pipeline output
    if dev:
        pipeline_dir = os.path.join(data_dir, 'pipeline_dev')
    else:
        pipeline_dir = os.path.join(data_dir, 'pipeline')
    util.create_directory_if_not_exists(pipeline_dir)

    # set up output directory
    output_dir = os.path.join(pipeline_dir, SCRIPT_NAME)
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if verbose: print(f'Set up output directory {output_dir}')

    # set up debugging directories
    if dev:
        debug_dir = os.path.join(os.path.dirname(output_dir), 'debug')
        util.create_directory_if_not_exists(debug_dir)
        util.clear_directory(debug_dir)
        if verbose: print(f'Set up debugging directory {debug_dir}')
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'max_recursion_limit'))
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'snr'))
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'masked_snr_arrays'))
    else:
        debug_dir = None

    # get psf cache directory
    psf_cache_dir = os.path.join(data_dir, 'cached_psfs')

    # get zeropoint magnitudes
    module_path = os.path.dirname(mejiro.__file__)
    zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

    # tuple the parameters
    tuple_list = []
    for run in range(runs):
        sca_id = str(scas[run % len(scas)]).zfill(2)
        sca_zp_dict = zp_dict[f'SCA{sca_id}']
        tuple_list.append((str(run).zfill(4), sca_id, sca_zp_dict, config,              output_dir, debug_dir, psf_cache_dir))

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # GalSim needs headroom
    process_count -= config['pipeline']['headroom_cores']['script_01']  # subtract off any additional cores
    count = runs
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = util.batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches, ascii=True):
        pool = Pool(processes=process_count)
        pool.map(run_slsim, batch)

    num_detectable = lens_util.count_detectable_lenses(output_dir)
    print(f'{num_detectable} detectable lenses found')
    print(f'{num_detectable / area / runs:.2f} per square degree')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME, os.path.join(pipeline_dir, 'execution_times.json'))


def run_slsim(tuple):
    # a legacy function but prevents duplicate runs
    np.random.seed()

    import mejiro
    from mejiro.helpers import survey_sim
    from mejiro.utils import util
    module_path = os.path.dirname(mejiro.__file__)

    # unpack tuple
    run, sca_id, sca_zp_dict, config, output_dir, debug_dir, psf_cache_dir = tuple

    # retrieve configuration parameters
    dev = config['pipeline']['dev']
    verbose = config['pipeline']['verbose']
    survey_config = config['pipeline']['survey']
    area = survey_config['area']
    bands = survey_config['bands']

    # prepare a directory for this particular run
    lens_output_dir = os.path.join(output_dir, f'run_{run}_sca{sca_id}')
    util.create_directory_if_not_exists(lens_output_dir)

    # set max recursion limit
    snr_input_num_pix = survey_config['snr_input_num_pix']
    sys.setrecursionlimit(snr_input_num_pix ** 2)

    # load SkyPy config file
    cache_dir = os.path.join(module_path, 'data', f'cached_skypy_configs_{area}')
    skypy_config = os.path.join(cache_dir,
                                f'roman_hlwas_sca{sca_id}.yml')  # TODO TEMP: there should be one source of truth for this, and if necessary, some code should update the cache behind the scenes
    # skypy_config = os.path.join(module_path, 'data', 'roman_hlwas.yml')
    config_file = util.load_skypy_config(skypy_config)
    if verbose: print(f'Loaded SkyPy configuration file {skypy_config}')

    # load Roman WFI filters
    roman_filters = sorted(glob(os.path.join(module_path, 'data', 'filter_responses', f'RomanSCA{sca_id}-*.ecsv')))
    _ = speclite.filters.load_filters(*roman_filters[:8])
    if verbose:
        print('Configured Roman filters. Loaded:')
        pprint(roman_filters[:8])

    # TODO save updated yaml file to cache

    # set HLWAS parameters
    config_file = util.load_skypy_config(skypy_config)  # read skypy config file to get survey area
    survey_area = float(config_file['fsky'][:-5])
    sky_area = Quantity(value=survey_area, unit='deg2')
    assert sky_area.value == area, f'Area mismatch: {sky_area.value} != {area}'
    cosmo = default_cosmology.get()  # TODO set cosmo in config file?
    if verbose: print(f'Surveying {sky_area.value} deg2 with bands {bands}')

    # define cuts on the intrinsic deflector and source populations (in addition to the skypy config file)
    kwargs_deflector_cut = {
        'band': survey_config['deflector_cut_band'],
        'band_max': survey_config['deflector_cut_band_max'],
        'z_min': survey_config['deflector_z_min'],
        'z_max': survey_config['deflector_z_max']
    }
    kwargs_source_cut = {
        'band': survey_config['source_cut_band'],
        'band_max': survey_config['source_cut_band_max'],
        'z_min': survey_config['source_z_min'],
        'z_max': survey_config['source_z_max']
    }

    # create the lens population
    if verbose: print('Defining galaxy population...')
    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=skypy_config,
        sky_area=sky_area,
        filters=None,
        cosmo=cosmo
    )
    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
    )
    lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    if verbose: print('Defined galaxy population')

    # draw the total lens population
    if survey_config['total_population']:
        if verbose: print('Identifying lenses...')
        kwargs_lens_total_cut = {
            'min_image_separation': 0,
            'max_image_separation': 10,
            'mag_arc_limit': None
        }
        total_lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_total_cut)
        if verbose: print(f'Number of total lenses: {len(total_lens_population)}')

        # compute SNRs and save
        if verbose: print(f'Computing SNRs for {len(total_lens_population)} lenses')
        snr_list = []
        num_exceptions = 0
        for candidate in tqdm(total_lens_population, disable=verbose, ascii=True):
            snr, _, _, _ = survey_sim.get_snr(candidate,
                                              cosmo=cosmo,
                                              band=survey_config['snr_band'],
                                              zp=sca_zp_dict[survey_config['snr_band']],
                                              detector=sca_id,
                                              detector_position=(2044, 2044),
                                              input_num_pix=survey_config['snr_input_num_pix'],
                                              output_num_pix=survey_config['snr_output_num_pix'],
                                              side=survey_config['snr_side'],
                                              oversample=survey_config['snr_oversample'],
                                              exposure_time=survey_config['snr_exposure_time'],
                                              add_subhalos=survey_config['snr_add_subhalos'],
                                              debugging=False,
                                              psf_cache_dir=psf_cache_dir)
            if snr is None:
                num_exceptions += 1
                continue

            snr_list.append(snr)
        np.save(os.path.join(output_dir, f'snr_list_{run}_sca{sca_id}.npy'), snr_list)

        # assert len(total_lens_population) == len(snr_list), f'Lengths of total_lens_population ({len(total_lens_population)}) and snr_list ({len(snr_list)}) do not match.'
        if verbose: print(f'Percentage of exceptions: {num_exceptions / len(total_lens_population) * 100:.2f}%')

        # save other params to CSV
        total_pop_csv = os.path.join(output_dir, f'total_pop_{run}_sca{sca_id}.csv')
        if verbose: print(f'Writing total population to {total_pop_csv}')
        survey_sim.write_lens_pop_to_csv(total_pop_csv, total_lens_population, snr_list, bands, verbose=verbose)

    # draw initial detectable lens population
    if verbose: print('Identifying detectable lenses...')
    kwargs_lens_detectable_cut = {
        'min_image_separation': survey_config['min_image_separation'],
        'max_image_separation': survey_config['max_image_separation'],
        'mag_arc_limit': {survey_config['mag_arc_limit_band']: survey_config['mag_arc_limit']}
    }
    lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_detectable_cut)
    if verbose: print(f'Number of detectable lenses from first set of criteria: {len(lens_population)}')

    # set up dict to capture some information about which candidates got filtered out
    if dev:
        filtered_sample = {}
        filtered_sample['total'] = len(lens_population)
        num_samples = 16
        filter_1, filter_2 = 0, 0
        filtered_sample['filter_1'] = []
        filtered_sample['filter_2'] = []

    # apply additional detectability criteria
    limit = None
    detectable_gglenses, detectable_snr_list, masked_snr_array_list = [], [], []
    k = 0
    for candidate in tqdm(lens_population, disable=not verbose, ascii=True):
        # Einstein radius and Sersic radius
        # _, kwargs_params = candidate.lenstronomy_kwargs(band=survey_params['large_lens_band'])
        # lens_mag = candidate.deflector_magnitude(band=survey_params['large_lens_band'])

        # if kwargs_params['kwargs_lens'][0]['theta_E'] < kwargs_params['kwargs_lens_light'][0][
        #     'R_sersic'] and lens_mag < survey_params['large_lens_mag_max']:
        #     filter_1 += 1
        #     if filter_1 <= num_samples:
        #         filtered_sample['filter_1'].append(candidate)
        #     continue

        # SNR
        snr, masked_snr_array, snr_list, _ = survey_sim.get_snr(candidate,
                                                                cosmo=cosmo,
                                                                band=survey_config['snr_band'],
                                                                zp=sca_zp_dict[survey_config['snr_band']],
                                                                detector=sca_id,
                                                                detector_position=(2044, 2044),
                                                                input_num_pix=survey_config['snr_input_num_pix'],
                                                                output_num_pix=survey_config['snr_output_num_pix'],
                                                                side=survey_config['snr_side'],
                                                                oversample=survey_config['snr_oversample'],
                                                                exposure_time=survey_config['snr_exposure_time'],
                                                                add_subhalos=survey_config['snr_add_subhalos'],
                                                                debugging=False,
                                                                debug_dir=debug_dir,
                                                                psf_cache_dir=psf_cache_dir)
        if snr is None:
            continue

        if snr < survey_config['snr_threshold']:
            # TODO filtering
            # if not debugging:
            #     filter_2 += 1
            #     if filter_2 <= num_samples:
            #         filtered_sample['filter_2'].append(candidate)
            continue

        # 1. extended source magnification
        extended_source_magnification = candidate.extended_source_magnification()[0]  # TODO confirm first element
        if snr < 50 and extended_source_magnification < survey_config['magnification']:
            # TODO filtering
            # if not debugging:
            #     filter_1 += 1
            #     if filter_1 <= num_samples:
            #         filtered_sample['filter_1'].append(candidate)
            continue

        # if both criteria satisfied, consider detectable
        detectable_gglenses.append(candidate)
        detectable_snr_list.append(snr)
        masked_snr_array_list.append(masked_snr_array)

        # if I've imposed a limit above this loop, exit the loop
        if limit is not None and len(detectable_gglenses) == limit:
            break

    if verbose: print(f'Run {run}: {len(detectable_gglenses)} detectable lens(es)')

    # TODO filtering
    # save information about which lenses got filtered out
    # if not debugging:  # TODO temp: make this configurable with its own option
    #     filtered_sample['num_filter_1'] = filter_1
    #     filtered_sample['num_filter_2'] = filter_2
    #     util.pickle(os.path.join(output_dir, f'filtered_sample_{run}_sca{sca_id}.pkl'), filtered_sample)

    assert len(detectable_gglenses) == len(
        detectable_snr_list), f'Lengths of detectable_gglenses ({len(detectable_gglenses)}) and detectable_snr_list ({len(detectable_snr_list)}) do not match.'

    if verbose: print('Retrieving lenstronomy parameters...')
    dict_list = []
    for gglens, snr, masked_snr_array in tqdm(zip(detectable_gglenses, detectable_snr_list, masked_snr_array_list),
                                              disable=not verbose, total=len(detectable_gglenses), ascii=True):

        # get lens params from gglens object
        kwargs_model, kwargs_params = gglens.lenstronomy_kwargs(
            band='F106')  # NB the band in arbitrary because all that changes is magnitude and we're overwriting that with the lens_mag and source_mag dicts below

        # build dicts for lens and source magnitudes
        lens_mags, source_mags, lensed_source_mags = {}, {}, {}
        for band in bands:
            lens_mags[band] = gglens.deflector_magnitude(band)
            source_mags[band] = gglens.extended_source_magnitude(band, lensed=False)[0]  # TODO confirm first element
            lensed_source_mags[band] = gglens.extended_source_magnitude(band, lensed=True)[0]  # TODO confirm first element

        z_lens, z_source = gglens.deflector_redshift, gglens.source_redshift_list[0]  # TODO confirm first element
        kwargs_lens = kwargs_params['kwargs_lens']

        # add additional necessary key/value pairs to kwargs_model
        kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
        kwargs_model['source_redshift_list'] = [z_source]
        kwargs_model['cosmo'] = cosmo
        kwargs_model['z_source'] = z_source
        kwargs_model['z_source_convention'] = survey_config['source_z_max']

        # create dict to pickle
        gglens_dict = {
            'kwargs_model': kwargs_model,
            'kwargs_params': kwargs_params,
            'lens_mags': lens_mags,
            'source_mags': source_mags,
            'lensed_source_mags': lensed_source_mags,
            'deflector_stellar_mass': gglens.deflector_stellar_mass(),
            'deflector_velocity_dispersion': gglens.deflector_velocity_dispersion(),
            'magnification': gglens.extended_source_magnification()[0],  # TODO confirm first element
            'snr': snr,
            'masked_snr_array': masked_snr_array,
            'num_images': len(gglens.point_source_image_positions()[0]),
            'sca': sca_id
        }

        pprint(gglens_dict)

        dict_list.append(gglens_dict)

    if len(dict_list) > 0:
        if verbose: print('Pickling lenses...')
        for i, each in tqdm(enumerate(dict_list), disable=not verbose, ascii=True):
            save_path = os.path.join(lens_output_dir, f'detectable_lens_{run}_sca{sca_id}_{str(i).zfill(5)}.pkl')
            util.pickle(save_path, each)

        detectable_pop_csv = os.path.join(output_dir, f'detectable_pop_{run}_sca{sca_id}.csv')
        survey_sim.write_lens_pop_to_csv(detectable_pop_csv, detectable_gglenses, detectable_snr_list, bands)

        detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{run}_sca{sca_id}.pkl')
        if verbose: print(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
        util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)
    else:
        if verbose: print(f'No detectable lenses found for run {run}, SCA{sca_id}')


if __name__ == '__main__':
    main()
