import argparse
import multiprocessing
import os
import time
import warnings
import yaml
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

import numpy as np
from astropy.cosmology import default_cosmology
from astropy.units import Quantity
from slsim.Lenses.lens_pop import LensPop
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors
from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline
from tqdm import tqdm

import mejiro
from mejiro.utils import util
from mejiro.analysis import snr_calculation
from mejiro.exposure import Exposure
from mejiro.galaxy_galaxy import GalaxyGalaxy
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import pipeline_util, roman_util, slsim_util, util


def main(args):
    start = time.time()

    # ensure the configuration file has a .yaml or .yml extension
    if not args.config.endswith(('.yaml', '.yml')):
        if os.path.exists(args.config + '.yaml'):
            args.config += '.yaml'
        elif os.path.exists(args.config + '.yml'):
            args.config += '.yml'
        else:
            raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")

    # read configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # set nice level
    os.nice(config.get('nice', 0))

    # suppress warnings
    if config['suppress_warnings']:
        warnings.filterwarnings("ignore", category=UserWarning)

    # retrieve configuration parameters
    dev = config['dev']
    verbose = config['verbose']
    data_dir = config['data_dir']
    psf_cache_dir = config['psf_cache_dir']
    runs = config['survey']['runs']
    detectors = config['survey']['detectors']
    area = config['survey']['area']

    # TODO TEMP: need to use PipelineHelper for this
    if hasattr(args, 'data_dir'):
        print(f'Overriding data_dir in config file ({data_dir}) with provided data_dir ({args.data_dir})')  # TODO logging
        data_dir = args.data_dir
    elif data_dir is None:
        raise ValueError("data_dir must be specified either in the config file or via the --data_dir argument.")

    # set configuration parameters
    config['survey']['cosmo'] = default_cosmology.get()
    if config['jaxtronomy']['use_jax']:
        os.environ['JAX_PLATFORM_NAME'] = config['jaxtronomy'].get('jax_platform', 'cpu')

    # load instrument
    instrument = pipeline_util.initialize_instrument_class(config['instrument'])

    # set up top directory for all pipeline output
    pipeline_dir = os.path.join(data_dir, config['pipeline_label'])
    if dev:
        pipeline_dir += '_dev'
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

    # set psf cache directory
    if psf_cache_dir is None:
        psf_cache_dir = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'psfs', instrument.name.lower())
    else:
        psf_cache_dir = os.path.join(data_dir, psf_cache_dir)

    # tuple the parameters
    tuple_list = []
    for run in range(runs):
        if instrument.num_detectors > 1:
            detector = detectors[run % len(detectors)]
        else:
            detector = None
        tuple_list.append((str(run).zfill(4), detector, config, output_dir, debug_dir, psf_cache_dir, instrument))

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = config['cores']['script_01']
    count = runs
    if count < process_count:
        process_count = count
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # process the tasks with ProcessPoolExecutor
    num_detectable = 0
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(run_slsim, task) for task in tuple_list]

        for future in tqdm(as_completed(futures), total=len(futures)):
            num_detectable += future.result()

    print(f'{num_detectable} detectable lenses found')
    print(f'{num_detectable / area / runs:.2f} per square degree')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, '01', os.path.join(pipeline_dir, 'execution_times.json'))


def run_slsim(tuple):
    module_path = os.path.dirname(mejiro.__file__)

    # unpack tuple
    run, detector, config, output_dir, debug_dir, psf_cache_dir, instrument = tuple

    # a legacy function but prevents duplicate runs
    np.random.seed()

    # retrieve configuration parameters
    limit = config['limit']
    snr_config = config['snr']
    verbose = config['verbose']
    use_jax = config['jaxtronomy']['use_jax']
    engine_params = config['imaging']['engine_params']
    survey_config = config['survey']
    area = survey_config['area']
    bands = survey_config['bands']
    cosmo = survey_config['cosmo']
    use_real_sources = survey_config['use_real_sources']
    use_slhammocks_pipeline = survey_config['use_slhammocks_pipeline']
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
    snr_detector = detector
    snr_detector_position = (2044, 2044)

    # set run identifier
    if instrument.name == 'Roman':
        run_id = f'{run}_{roman_util.get_sca_string(detector).lower()}'
    elif instrument.name == 'HWO':
        run_id = str(run).zfill(4)
    else:
        raise ValueError(f"Run identifier not implemented for {instrument.name}.")

    # load filters
    if instrument.name == 'Roman':
        detector_string = roman_util.get_sca_string(detector).lower()
        filter_args = {'detector': detector_string}
    elif instrument.name == 'HWO':
        filter_args = {}
    else:
        raise ValueError(f"Speclite filter loading not implemented for {instrument.name}.")
    speclite_filters = instrument.load_speclite_filters(**filter_args)
    if verbose: print(f'Loaded {instrument.name} filter response curve(s): {speclite_filters.names}')

    # load SkyPy config file
    cache_dir = os.path.join(module_path, 'data', 'skypy', config['survey']['skypy_config'])
    if instrument.name == 'Roman':
        skypy_config = os.path.join(cache_dir,
                                f'{config["survey"]["skypy_config"]}_{detector_string}.yml')  # TODO TEMP: there should be one source of truth for this, and if necessary, some code should update the cache behind the scenes
    elif instrument.name == 'HWO':
        skypy_config = os.path.join(cache_dir, config['survey']['skypy_config'] + '.yml')
    else:
        raise ValueError(f"SkyPy configuration file retrieval not implemented for {instrument.name}.")
    config_file = util.load_skypy_config(skypy_config)  # read skypy config file to get survey area
    if verbose: print(f'Loaded SkyPy configuration file {skypy_config}')

    # load SLHammocks SkyPy config file
    if use_slhammocks_pipeline:
        slhammocks_pipeline_kwargs = survey_config['slhammocks_pipeline_kwargs']
        cache_dir = os.path.join(module_path, 'data', 'skypy', slhammocks_pipeline_kwargs["skypy_config"])
        if instrument.name == 'Roman':
            slhammocks_config = os.path.join(cache_dir,
                                    f'{slhammocks_pipeline_kwargs["skypy_config"]}_{detector_string}.yml')  # TODO TEMP: there should be one source of truth for this, and if necessary, some code should update the cache behind the scenes
        slhammocks_pipeline_kwargs["skypy_config"] = slhammocks_config
        if verbose: print(f'Loaded SLHammocks configuration file {slhammocks_config}')

    # set survey parameters
    survey_area = float(config_file['fsky'][:-5])
    sky_area = Quantity(value=survey_area, unit='deg2')
    assert sky_area.value == area, f'Area mismatch: {sky_area.value} != {area}'
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
    if use_slhammocks_pipeline:
        halo_galaxy_pipeline = SLHammocksPipeline(
            sky_area=sky_area,
            cosmo=cosmo,
            z_min=survey_config['deflector_z_min'],
            z_max=survey_config['deflector_z_max'],
            **slhammocks_pipeline_kwargs
        )
        lens_galaxies = deflectors.CompoundLensHalosGalaxies(
            halo_galaxy_list=halo_galaxy_pipeline.halo_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light={},
            cosmo=cosmo,
            sky_area=sky_area,
        )
    else:
        lens_galaxies = deflectors.AllLensGalaxies(
            red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
            blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light={},
            cosmo=cosmo,
            sky_area=sky_area,
        )
    real_galaxy_kwargs = {
        "extended_source_type": "catalog_source",
        "extendedsource_kwargs": survey_config["catalog_source_kwargs"]
    } if use_real_sources else {}
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        source_size=None,
        **real_galaxy_kwargs
    )
    lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    if verbose: print('Defined galaxy population')

    # get PSF parameters for SNR calculation
    instrument_psf_kwargs = {
        'band': snr_band,
        'detector': snr_detector,
        'detector_position': snr_detector_position,
        'oversample': 5,
        'num_pix': 101,
        'check_cache': True,
        'psf_cache_dir': psf_cache_dir,
        'verbose': False
    }
    kwargs_psf = instrument.get_psf_kwargs(**instrument_psf_kwargs)

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
        for candidate in tqdm(total_lens_population, disable=verbose):
            strong_lens = GalaxyGalaxy.from_slsim(candidate, bands=bands, use_jax=use_jax)

            # TODO temporary fix to make sure that there are two images formed
            image_positions = strong_lens.get_image_positions()
            if len(image_positions[0]) < 2:
                continue

            # TODO do something with the substract lens flag
            # TODO do something with the add subhalos flag

            # calculate SNR
            synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                            instrument=instrument,
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
                                engine_params=engine_params,
                                verbose=False)
            snr, _ = snr_calculation.get_snr(exposure=exposure,
                                            snr_per_pixel_threshold=snr_per_pixel_threshold,
                                            verbose=False)
            snr_list.append(snr)
            
            if snr is None:
                num_exceptions += 1
            
        if verbose:
            if len(total_lens_population) > 0:
                print(f'Percentage of exceptions: {num_exceptions / len(total_lens_population) * 100:.2f}%')
            else:
                warnings.warn('No systems in total population. Consider revising the galaxy population.')
                return 0

        # save other params to CSV
        if survey_config["write_to_csv"]:
            total_pop_csv = os.path.join(output_dir, f'total_pop_{run_id}.csv')
            if verbose: print(f'Writing total population to {total_pop_csv}')
            slsim_util.write_lens_population_to_csv(total_pop_csv, total_lens_population, snr_list, bands=bands, verbose=verbose)

    # draw initial detectable lens population
    if verbose: print('Identifying detectable lenses...')
    kwargs_lens_detectable_cut = {
        'min_image_separation': survey_config['min_image_separation'],
        'max_image_separation': survey_config['max_image_separation'],
        'mag_arc_limit': {survey_config['mag_arc_limit_band']: survey_config['mag_arc_limit']}
    }
    lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_detectable_cut)
    if verbose: print(f'Number of detectable lenses from first set of criteria: {len(lens_population)}')

    # apply additional detectability criteria
    detectable_gglenses, detectable_snr_list, masked_snr_array_list = [], [], []
    for candidate in tqdm(lens_population, disable=not verbose):
        # convert from SLSim gglens to mejiro GalaxyGalaxy
        try:
            strong_lens = GalaxyGalaxy.from_slsim(candidate, bands=bands, use_jax=use_jax)
        except:
            print(f'Error processing candidate {candidate["name"]}')
            continue

        # TODO do something with the substract lens flag
        # TODO do something with the add subhalos flag

        # criterion 1: SNR
        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                        instrument=instrument,
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
                            engine_params=engine_params,
                            verbose=False)
        snr, masked_snr_array = snr_calculation.get_snr(exposure=exposure,
                                                        snr_per_pixel_threshold=snr_per_pixel_threshold,
                                                        verbose=False)
        if snr is None or snr < snr_config['snr_threshold']:
            continue

        # criterion 2: extended source magnification
        magnification = strong_lens.physical_params['magnification']
        if magnification < survey_config['magnification']:
            continue

        # if both criteria satisfied, consider detectable
        detectable_gglenses.append(candidate)
        detectable_snr_list.append(snr)
        masked_snr_array_list.append(masked_snr_array)

        # if I've imposed a limit above this loop, exit the loop
        if limit is not None and len(detectable_gglenses) == limit:
            break

    if verbose: print(f'Run {run}: {len(detectable_gglenses)} detectable lens(es)')

    assert len(detectable_gglenses) == len(
        detectable_snr_list), f'Lengths of detectable_gglenses ({len(detectable_gglenses)}) and detectable_snr_list ({len(detectable_snr_list)}) do not match.'

    if len(detectable_gglenses) > 0:
        # pickle the list of slsim gglenses
        detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{run_id}.pkl')
        if verbose: print(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
        util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)

        # write the parameters of detectable lenses to CSV
        if survey_config["write_to_csv"]:
            detectable_pop_csv = os.path.join(output_dir, f'detectable_pop_{run_id}.csv')
            slsim_util.write_lens_population_to_csv(detectable_pop_csv, detectable_gglenses, detectable_snr_list, bands=bands, verbose=verbose)
    else:
        if verbose: print(f'No detectable lenses found for run {run}')
    
    return len(detectable_gglenses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate survey and identify detectable systems")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False, help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    args = parser.parse_args()
    main(args)
