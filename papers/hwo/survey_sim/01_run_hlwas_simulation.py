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


def main():
    start = time.time()

    # read configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    repo_dir = config['repo_dir']

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util
    from mejiro.instruments.hwo import HWO

    # set nice level
    os.nice(config['nice'])

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    runs = config['survey']['runs']
    area = config['survey']['area']

    # set up top directory for all pipeline output
    if dev:
        pipeline_dir = os.path.join(data_dir, 'hwo_survey_dev')
    else:
        pipeline_dir = os.path.join(data_dir, 'hwo_survey')
    util.create_directory_if_not_exists(pipeline_dir)

    # set up output directory
    output_dir = os.path.join(pipeline_dir, '01')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)
    if verbose: print(f'Set up output directory {output_dir}')

    # set up debugging directories
    if dev:
        debug_dir = os.path.join(os.path.dirname(output_dir), 'debug')
        util.create_directory_if_not_exists(debug_dir)
        util.clear_directory(debug_dir)
        if verbose: print(f'Set up debugging directory {debug_dir}')
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'snr'))
        util.create_directory_if_not_exists(os.path.join(debug_dir, 'masked_snr_arrays'))
    else:
        debug_dir = None

    # load instrument
    hwo = HWO()

    # tuple the parameters
    tuple_list = []
    for run in range(runs):
        tuple_list.append((str(run).zfill(4), config, output_dir, debug_dir, hwo))

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = int(cpu_count / 2)  # GalSim needs headroom
    process_count -= config['headroom_cores']['script_01']  # subtract off any additional cores
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

    num_detectable = len(glob(output_dir + '/**/detectable_lens_*.pkl'))
    print(f'{num_detectable} detectable lenses found')
    print(f'{num_detectable / area / runs:.2f} per square degree')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '01', os.path.join(pipeline_dir, 'execution_times.json'))


def run_slsim(tuple):
    # a legacy function but prevents duplicate runs
    np.random.seed()

    import mejiro
    from mejiro.analysis import snr_calculation
    from mejiro.exposure import Exposure
    from mejiro.engines.stpsf_engine import STPSFEngine
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils import slsim_util, util
    module_path = os.path.dirname(mejiro.__file__)

    # unpack tuple
    run, config, output_dir, debug_dir, hwo = tuple

    # retrieve configuration parameters
    dev = config['dev']
    snr_config = config['snr']
    verbose = config['verbose']
    survey_config = config['survey']
    area = survey_config['area']
    bands = survey_config['bands']
    snr_band = snr_config['snr_band']
    snr_exposure_time = snr_config['snr_exposure_time']
    snr_fov_arcsec = snr_config['snr_fov_arcsec']
    snr_supersampling_factor = snr_config['snr_supersampling_factor']
    snr_supersampling_compute_mode = snr_config['snr_supersampling_compute_mode']
    snr_per_pixel_threshold = snr_config['snr_per_pixel_threshold']
    snr_kwargs_numerics = {
        'supersampling_factor': snr_supersampling_factor,
        'compute_mode': snr_supersampling_compute_mode,
    }

    # prepare a directory for this particular run
    lens_output_dir = os.path.join(output_dir, f'run_{run}')
    util.create_directory_if_not_exists(lens_output_dir)

    # load HWO HRI filters
    hwo_filters = hwo.load_speclite_filters()
    if verbose:
        print('Configured HWO filters. Loaded:')
        pprint(hwo_filters)

    # load SkyPy config file
    skypy_config = os.path.join(module_path, 'data', 'hwo.yml')
    config_file = util.load_skypy_config(skypy_config)  # read skypy config file to get survey area
    if verbose: print(f'Loaded SkyPy configuration file {skypy_config}')

    # set HLWAS parameters
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
            strong_lens = slsim_util.slsim_lens_to_mejiro(slsim_lens=candidate,
                                                          bands=[snr_band],
                                                          cosmo=cosmo)
            
            kwargs_psf = STPSFEngine.get_roman_psf_kwargs(snr_band, snr_detector, snr_detector_position, oversample=snr_supersampling_factor, num_pix=101, check_cache=True, psf_cache_dir=psf_cache_dir, verbose=False)

            # TODO do something with the substract lens flag
            # TODO do something with the add subhalos flag

            synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                            instrument=roman,
                                            band=snr_band,
                                            fov_arcsec=snr_fov_arcsec,
                                            instrument_params={'detector': snr_detector, 'detector_position': snr_detector_position},
                                            kwargs_numerics=snr_kwargs_numerics,
                                            kwargs_psf=kwargs_psf,
                                            pieces=True,
                                            verbose=False)

            exposure = Exposure(synthetic_image=synthetic_image,
                                exposure_time=snr_exposure_time,
                                engine='galsim',
                                verbose=False)

            snr, _ = snr_calculation.get_snr(exposure=exposure,
                                            snr_per_pixel_threshold=snr_per_pixel_threshold,
                                            verbose=False)
            if snr is None:
                num_exceptions += 1
                continue

            snr_list.append(snr)
        np.save(os.path.join(output_dir, f'snr_list_{run}.npy'), snr_list)

        if verbose: print(f'Percentage of exceptions: {num_exceptions / len(total_lens_population) * 100:.2f}%')

        # save other params to CSV
        total_pop_csv = os.path.join(output_dir, f'total_pop_{run}.csv')
        if verbose: print(f'Writing total population to {total_pop_csv}')
        slsim_util.write_lens_pop_to_csv(total_pop_csv, total_lens_population, snr_list, bands, verbose=verbose)

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
    # if dev:
    #     filtered_sample = {}
    #     filtered_sample['total'] = len(lens_population)
    #     num_samples = 16
    #     filter_1, filter_2 = 0, 0
    #     filtered_sample['filter_1'] = []
    #     filtered_sample['filter_2'] = []

    # apply additional detectability criteria
    limit = config['limit']
    detectable_gglenses, detectable_snr_list, masked_snr_array_list = [], [], []
    k = 0
    for candidate in tqdm(lens_population, disable=not verbose, ascii=True):
        # criterion 1: SNR
        strong_lens = slsim_util.slsim_lens_to_mejiro(slsim_lens=candidate,
                                                          bands=[snr_band],
                                                          cosmo=cosmo)
            
        kwargs_psf = STPSFEngine.get_roman_psf_kwargs(snr_band, snr_detector, snr_detector_position, oversample=snr_supersampling_factor, num_pix=101, check_cache=True, psf_cache_dir=psf_cache_dir, verbose=False)

        # TODO do something with the substract lens flag
        # TODO do something with the add subhalos flag

        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                        instrument=roman,
                                        band=snr_band,
                                        fov_arcsec=snr_fov_arcsec,
                                        instrument_params={'detector': snr_detector, 'detector_position': snr_detector_position},
                                        kwargs_numerics=snr_kwargs_numerics,
                                        kwargs_psf=kwargs_psf,
                                        pieces=True,
                                        verbose=False)

        exposure = Exposure(synthetic_image=synthetic_image,
                            exposure_time=snr_exposure_time,
                            engine='galsim',
                            verbose=False)

        snr, masked_snr_array = snr_calculation.get_snr(exposure=exposure,
                                                        snr_per_pixel_threshold=snr_per_pixel_threshold,
                                                        verbose=False)
        if snr is None:
            continue

        if snr < snr_config['snr_threshold']:
            # TODO filtering
            # if not debugging:
            #     filter_2 += 1
            #     if filter_2 <= num_samples:
            #         filtered_sample['filter_2'].append(candidate)
            continue

        # criterion 2: extended source magnification
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

    if len(detectable_gglenses) > 0:
        if verbose: print('Pickling lenses...')
        for i, each in tqdm(enumerate(detectable_gglenses), disable=not verbose, ascii=True):
            save_path = os.path.join(lens_output_dir, f'detectable_lens_{run}_sca{sca_id}_{str(i).zfill(5)}.pkl')
            util.pickle(save_path, each)

        detectable_pop_csv = os.path.join(output_dir, f'detectable_pop_{run}_sca{sca_id}.csv')
        slsim_util.write_lens_pop_to_csv(detectable_pop_csv, detectable_gglenses, detectable_snr_list, bands, verbose=verbose)

        detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{run}_sca{sca_id}.pkl')
        if verbose: print(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
        util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)
    else:
        if verbose: print(f'No detectable lenses found for run {run}, SCA{sca_id}')


if __name__ == '__main__':
    main()
