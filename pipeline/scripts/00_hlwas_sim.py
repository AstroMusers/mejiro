import json
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
from tqdm import tqdm


@hydra.main(version_base=None, config_path='../../config', config_name='config.yaml')
def main(config):
    start = time.time()

    # enable use of local packages
    repo_dir = config.machine.repo_dir
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    import mejiro
    from mejiro.utils import util
    from mejiro.lenses import lens_util

    # retrieve configuration parameters
    pipeline_params = util.hydra_to_dict(config.pipeline)
    survey_params = util.hydra_to_dict(config.survey)
    debugging = pipeline_params['debugging']

    # set up output directory
    if debugging:
        output_dir = os.path.join(f'{config.machine.pipeline_dir}_dev',
                                  '00')  # TODO eventually, this should do something like reading the dir_00 from the config file, taking the parent dir, appending _dev, then appending whatever the basepath of dir_00 is: otherwise, there's no point in allowing dir_00 to be configurable. also, this change must take place for all scripts
    else:
        output_dir = config.machine.dir_00
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if debugging: print(f'Set up output directory {output_dir}')

    # set up debugging directories
    if debugging:
        debug_dir = os.path.join(os.path.dirname(output_dir), 'debug')
        util.create_directory_if_not_exists(debug_dir)
        util.clear_directory(debug_dir)
        print(f'Set up debugging directory {debug_dir}')
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'max_recursion_limit'))
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'snr'))
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'masked_snr_arrays'))
    else:
        debug_dir = None

    # get psf cache directory
    psf_cache_dir = os.path.join(config.machine.data_dir, 'cached_psfs')

    # get zeropoint magnitudes
    module_path = os.path.dirname(mejiro.__file__)
    zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

    # tuple the parameters
    runs = survey_params['runs']
    scas = survey_params['scas']
    area = survey_params['area']
    tuple_list = []
    for run in range(runs):
        sca_id = str(scas[run % len(scas)]).zfill(2)
        sca_zp_dict = zp_dict[f'SCA{sca_id}']
        tuple_list.append((str(run).zfill(4), sca_id, sca_zp_dict, area, survey_params, pipeline_params,
                           output_dir, debugging, debug_dir, psf_cache_dir))

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count - config.machine.headroom_cores
    process_count -= int(cpu_count / 2)  # GalSim needs headroom
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

    num_detectable = lens_util.count_detectable_lenses(output_dir)
    print(f'{num_detectable} detectable lenses found')
    print(f'{num_detectable / area / runs:.2f} per square degree')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '00', os.path.join(os.path.dirname(output_dir), 'execution_times.json'))


def run_slsim(tuple):
    # a legacy function but prevents duplicate runs
    np.random.seed()

    import mejiro
    from mejiro.helpers import survey_sim
    from mejiro.utils import util

    module_path = os.path.dirname(mejiro.__file__)

    # unpack tuple
    run, sca_id, sca_zp_dict, area, survey_params, pipeline_params, output_dir, debugging, debug_dir, psf_cache_dir = tuple

    # prepare a directory for this particular run
    lens_output_dir = os.path.join(output_dir, f'run_{run}_sca{sca_id}')
    util.create_directory_if_not_exists(lens_output_dir)

    # set max recursion limit
    sys.setrecursionlimit(survey_params['snr_num_pix'] ** 2)

    # load SkyPy config file
    cache_dir = os.path.join(module_path, 'data', f'cached_skypy_configs_{area}')
    skypy_config = os.path.join(cache_dir,
                                f'roman_hlwas_sca{sca_id}.yml')  # TODO TEMP: there should be one source of truth for this, and if necessary, some code should update the cache behind the scenes
    # skypy_config = os.path.join(module_path, 'data', 'roman_hlwas.yml')
    config_file = util.load_skypy_config(skypy_config)
    if debugging: print(f'Loaded SkyPy configuration file {skypy_config}')

    # load Roman WFI filters
    roman_filters = sorted(glob(os.path.join(module_path, 'data', 'filter_responses', f'RomanSCA{sca_id}-*.ecsv')))
    # configure_roman_filters()
    # roman_filters = filter_names()
    # roman_filters.sort()
    _ = speclite.filters.load_filters(*roman_filters[:8])
    if debugging:
        print('Configured Roman filters. Loaded:')
        pprint(roman_filters[:8])

    # TODO save updated yaml file to cache

    # set HLWAS parameters
    config_file = util.load_skypy_config(skypy_config)  # read skypy config file to get survey area
    survey_area = float(config_file['fsky'][:-5])
    sky_area = Quantity(value=survey_area, unit='deg2')
    assert sky_area.value == area, f'Area mismatch: {sky_area.value} != {area}'  # TODO temp
    cosmo = default_cosmology.get()
    # from astropy.cosmology import FlatLambdaCDM
    # cosmo = FlatLambdaCDM(H0=67.66, Om0=0.30966, Ob0=0.04897)
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
    if survey_params['total_population']:
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
        # j = 0
        num_exceptions = 0
        for candidate in tqdm(total_lens_population, disable=not debugging):
            snr, _, _, _ = survey_sim.get_snr(candidate,
                                              band=survey_params['snr_band'],
                                              zp=sca_zp_dict[survey_params['snr_band']],
                                              detector=sca_id,
                                              detector_position=(2048, 2048),
                                              num_pix=survey_params['snr_num_pix'],
                                              side=survey_params['snr_side'],
                                              oversample=survey_params['snr_oversample'],
                                              exposure_time=survey_params['snr_exposure_time'],
                                              add_subhalos=survey_params['snr_add_subhalos'],
                                              debugging=False,
                                              psf_cache_dir=psf_cache_dir)
            if snr is None:
                num_exceptions += 1
                continue

            snr_list.append(snr)
            # if debugging:
            #     if j < 25:
            #         util.pickle(os.path.join(output_dir, f'masked_snr_array_{str(j).zfill(8)}.pkl'), masked_snr_array)
            #         util.pickle(os.path.join(output_dir, f'masked_snr_array_snr_{str(j).zfill(8)}.pkl'), snr)
            # j += 1
        np.save(os.path.join(output_dir, f'snr_list_{run}_sca{sca_id}.npy'), snr_list)

        if debugging:
            print(f'Number of exceptions: {num_exceptions}; {num_exceptions / len(snr_list) * 100:.2f}%')

        # TODO once get_regions issue(s) resolved, the following line should be reinstated
        # assert len(total_lens_population) == len(snr_list), f'Lengths of total_lens_population ({len(total_lens_population)}) and snr_list ({len(snr_list)}) do not match.'

        # save other params to CSV
        total_pop_csv = os.path.join(output_dir, f'total_pop_{run}_sca{sca_id}.csv')
        if debugging: print(f'Writing total population to {total_pop_csv}')
        survey_sim.write_lens_pop_to_csv(total_pop_csv, total_lens_population, snr_list, bands,
                                         suppress_output=not debugging)

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
    detectable_gglenses, detectable_snr_list, masked_snr_array_list = [], [], []
    k = 0
    for candidate in tqdm(lens_population, disable=not debugging):
        # 1. Einstein radius and Sersic radius
        # _, kwargs_params = candidate.lenstronomy_kwargs(band=survey_params['large_lens_band'])
        # lens_mag = candidate.deflector_magnitude(band=survey_params['large_lens_band'])

        # if kwargs_params['kwargs_lens'][0]['theta_E'] < kwargs_params['kwargs_lens_light'][0][
        #     'R_sersic'] and lens_mag < survey_params['large_lens_mag_max']:
        #     filter_1 += 1
        #     if filter_1 <= num_samples:
        #         filtered_sample['filter_1'].append(candidate)
        #     continue

        # 1. extended source magnification
        # extended_source_magnification = candidate.extended_source_magnification()
        # if extended_source_magnification < survey_params['magnification']:
        #     filter_1 += 1
        #     if filter_1 <= num_samples:
        #         filtered_sample['filter_1'].append(candidate)
        #     continue

        # 2. SNR
        snr, masked_snr_array, snr_list, _ = survey_sim.get_snr(candidate,
                                                                band=survey_params['snr_band'],
                                                                zp=sca_zp_dict[survey_params['snr_band']],
                                                                detector=sca_id,
                                                                detector_position=(2048, 2048),
                                                                num_pix=survey_params['snr_num_pix'],
                                                                side=survey_params['snr_side'],
                                                                oversample=survey_params['snr_oversample'],
                                                                exposure_time=survey_params['snr_exposure_time'],
                                                                add_subhalos=survey_params['snr_add_subhalos'],
                                                                debugging=debugging,
                                                                debug_dir=debug_dir,
                                                                psf_cache_dir=psf_cache_dir)
        if snr is None:
            continue

        # if debugging and k % 100 == 0:
        #     plt.imshow(masked_snr_array)
        #     plt.title(f'SNR: {snr}, SNR list: {snr_list}')
        #     plt.colorbar()
        #     plt.savefig(os.path.join(debug_dir, 'masked_snr_arrays', f'masked_snr_array_{id(masked_snr_array)}.png'))
        #     plt.close()
        # k += 1

        if snr < survey_params['snr_threshold']:
            # filter this candidate out
            filter_2 += 1
            if filter_2 <= num_samples:
                filtered_sample['filter_2'].append(candidate)
            continue

        # if both criteria satisfied, consider detectable
        detectable_gglenses.append(candidate)
        detectable_snr_list.append(snr)
        masked_snr_array_list.append(masked_snr_array)

        # if I've imposed a limit above this loop, exit the loop
        if limit is not None and len(detectable_gglenses) == limit:
            break

    if debugging: print(f'Run {run}: {len(detectable_gglenses)} detectable lens(es)')

    # save information about which lenses got filtered out
    filtered_sample['num_filter_1'] = filter_1
    filtered_sample['num_filter_2'] = filter_2
    # util.pickle(os.path.join(output_dir, f'filtered_sample_{run}_sca{sca_id}.pkl'), filtered_sample)  # TODO temp: make this configurable

    assert len(detectable_gglenses) == len(
        detectable_snr_list), f'Lengths of detectable_gglenses ({len(detectable_gglenses)}) and detectable_snr_list ({len(detectable_snr_list)}) do not match.'

    if debugging: print('Retrieving lenstronomy parameters...')
    dict_list = []
    for gglens, snr, masked_snr_array in tqdm(zip(detectable_gglenses, detectable_snr_list, masked_snr_array_list),
                                              disable=not debugging, total=len(detectable_gglenses)):

        # get lens params from gglens object
        kwargs_model, kwargs_params = gglens.lenstronomy_kwargs(
            band='F106')  # NB the band in arbitrary because all that changes is magnitude and we're overwriting that with the lens_mag and source_mag dicts below

        # build dicts for lens and source magnitudes
        lens_mags, source_mags, lensed_source_mags = {}, {}, {}
        for band in bands:
            lens_mags[band] = gglens.deflector_magnitude(band)
            source_mags[band] = gglens.extended_source_magnitude(band, lensed=False)
            lensed_source_mags[band] = gglens.extended_source_magnitude(band, lensed=True)

        z_lens, z_source = gglens.deflector_redshift, gglens.source_redshift
        kwargs_lens = kwargs_params['kwargs_lens']

        # add additional necessary key/value pairs to kwargs_model
        kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
        kwargs_model['source_redshift_list'] = [z_source]
        kwargs_model['cosmo'] = cosmo
        kwargs_model['z_source'] = z_source
        kwargs_model['z_source_convention'] = survey_params['source_z_max']

        # create dict to pickle
        gglens_dict = {
            'kwargs_model': kwargs_model,
            'kwargs_params': kwargs_params,
            'lens_mags': lens_mags,
            'source_mags': source_mags,
            'lensed_source_mags': lensed_source_mags,
            'deflector_stellar_mass': gglens.deflector_stellar_mass(),
            'deflector_velocity_dispersion': gglens.deflector_velocity_dispersion(),
            'magnification': gglens.extended_source_magnification(),
            'snr': snr,
            'masked_snr_array': masked_snr_array,
            'sca': sca_id
        }

        dict_list.append(gglens_dict)

    if len(dict_list) > 0:
        if debugging: print('Pickling lenses...')
        for i, each in tqdm(enumerate(dict_list), disable=not debugging):
            save_path = os.path.join(lens_output_dir, f'detectable_lens_{run}_sca{sca_id}_{str(i).zfill(5)}.pkl')
            util.pickle(save_path, each)

        detectable_pop_csv = os.path.join(output_dir, f'detectable_pop_{run}_sca{sca_id}.csv')
        survey_sim.write_lens_pop_to_csv(detectable_pop_csv, detectable_gglenses, detectable_snr_list, bands)

        detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{run}_sca{sca_id}.pkl')
        if debugging: print(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
        util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)
    else:
        if debugging: print(f'No detectable lenses found for run {run}, SCA{sca_id}')


if __name__ == '__main__':
    main()
