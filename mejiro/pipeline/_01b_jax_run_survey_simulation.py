"""
JAX-accelerated variant of ``_01b_run_survey_simulation.py``.

Routes lenstronomy ray-shooting (invoked during per-candidate SNR computation)
through jaxtronomy by flipping ``StrongLens.use_jax`` to True on every profile
in ``_JAXXED_MODELS`` after building each ``GalaxyGalaxy`` via ``from_slsim``.

Execution topology is selected from the YAML config's ``jaxtronomy.jax_platform``:

* ``cpu`` -- ``ProcessPoolExecutor`` with a **spawn** multiprocessing context.
  Each worker imports JAX after ``JAX_PLATFORM_NAME`` is set, and all workers
  share an on-disk JAX compilation cache so the JIT cost for each unique
  ``lens_model_list`` signature is paid only once across the whole run.

* ``gpu`` -- single-process sequential execution over runs. One in-flight
  kernel per GPU; oversubscribing N processes onto one device thrashes memory.

Worker threading (CPU path):
    Three layered controls keep N workers from collectively oversubscribing
    the host:

    1. BLAS/OMP env vars (``OMP_NUM_THREADS`` etc.) set below before numpy
       imports, so NumPy/SciPy/galsim stay single-threaded in every worker.
    2. ``XLA_FLAGS`` set in ``_worker_init`` (unconditional assignment, not
       setdefault, so an inherited shell value cannot mask it):
       ``--xla_cpu_multi_thread_eigen=false`` disables Eigen op-level
       parallelism; ``--xla_force_host_platform_device_count=1`` keeps JAX
       to one logical CPU device per worker.
    3. ``os.sched_setaffinity`` in ``_worker_init`` pins each worker to a
       single core (ids handed out via a ``multiprocessing.Queue`` built in
       ``_run_spawn_pool``). This is the actual hard cap: JAX's CPU PJRT
       client still allocates dispatch and JIT-compile threads sized to
       ``hardware_concurrency()`` that ``XLA_FLAGS`` does not bound, but
       affinity restricts them to one core so the system run queue stays
       near ``workers`` rather than ``workers * ~3``.

Outputs land in ``<pipeline_dir>/01b_jax/`` with the same naming convention as
step 01b, so downstream steps can consume either step 01b or step 01b_jax
outputs.

Usage:
    python3 _01b_jax_run_survey_simulation.py --config <config.yaml> [--data_dir <output_dir>] [--resume]

Config additions (optional):
    cores.script_01b_jax: <int>   # falls back to cores.script_01b if absent
"""
import os

# Single-thread BLAS/OMP in every worker; must happen before numpy imports.
# JAX/XLA threading and per-worker CPU affinity are handled later in
# ``_worker_init`` -- see module docstring "Worker threading".
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import logging
import multiprocessing
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from astropy.cosmology import default_cosmology
from astropy.units import Quantity
from slsim.Lenses.lens_pop import LensPop
import slsim.Sources as sources
import slsim.Deflectors as deflectors
from tqdm import tqdm

from mejiro.analysis import snr_calculation
from mejiro.exposure import Exposure
from mejiro.galaxy_galaxy import GalaxyGalaxy
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import roman_util, slsim_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '01a'
SCRIPT_NAME = '01b_jax'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']

# Populated per-worker by ``_worker_init`` so run_slsim can flip use_jax without
# re-importing jaxtronomy for every lens.
_JAXXED_MODELS_SET = None


def main(args):
    start = time.time()

    # initialize PipelineHelper (we handle the default wipe ourselves so we can count + warn first)
    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS, delete_existing_output=False)

    if not args.resume:
        existing = glob(os.path.join(pipeline.output_dir, '*'))
        if existing:
            logger.warning(
                f'Deleting {len(existing)} existing output file(s) in '
                f'{pipeline.output_dir} and rebuilding from scratch. Pass --resume to keep them.'
            )
            util.clear_directory(pipeline.output_dir)

    # set configuration parameters
    pipeline.config['survey']['cosmo'] = default_cosmology.get()

    jax_platform = pipeline.config.get('jaxtronomy', {}).get('jax_platform', 'cpu')

    # Verify JAX is importable and has devices for the requested platform. Run
    # in a subprocess so the main process's JAX platform stays unlocked; we
    # never import jax in this module.
    _verify_jax_platform(jax_platform)

    # Co-locate the persistent JIT cache with the outputs so it survives across
    # runs and is shared between all spawn workers.
    compilation_cache_dir = os.path.join(pipeline.output_dir, '.jax_cache')
    os.makedirs(compilation_cache_dir, exist_ok=True)

    # discover pre-computed galaxy tables from _01a
    galaxy_table_paths = sorted(glob(os.path.join(pipeline.input_dir, 'galaxy_table_*.pkl')))
    num_galaxy_tables = len(galaxy_table_paths)
    if num_galaxy_tables == 0:
        raise FileNotFoundError(f'No galaxy tables found in {pipeline.input_dir}. Run _01a first.')
    logger.info(f'Found {num_galaxy_tables} pre-computed galaxy table(s) in {pipeline.input_dir}')

    # tuple the parameters
    tuple_list = []
    for run in range(pipeline.runs):
        if pipeline.instrument.num_detectors > 1:
            detector = pipeline.detectors[run % len(pipeline.detectors)]
        else:
            detector = None
        table_path = galaxy_table_paths[run % num_galaxy_tables]
        tuple_list.append((str(run).zfill(4), detector, table_path, pipeline.config,
                           pipeline.output_dir, pipeline.psf_cache_dir, pipeline.instrument))

    # check for already-completed runs from a previous (incomplete) execution
    def _run_id(run_str, detector):
        if pipeline.instrument.name == 'Roman':
            return f'{run_str}_{roman_util.get_sca_string(detector).lower()}'
        return run_str

    num_detectable = 0
    skipped = 0
    filtered_tuple_list = []
    for params in tuple_list:
        run_str, detector = params[0], params[1]
        rid = _run_id(run_str, detector)
        sentinel = os.path.join(pipeline.output_dir, f'run_complete_{rid}.txt')
        if os.path.exists(sentinel):
            with open(sentinel) as f:
                num_detectable += int(f.read().strip())
            skipped += 1
        else:
            filtered_tuple_list.append(params)

    if args.resume:
        logger.info(
            f'Resuming: {skipped} of {len(tuple_list)} run(s) already complete, '
            f'{len(filtered_tuple_list)} remaining.'
        )

    if not filtered_tuple_list:
        logger.info('All runs already completed. Nothing to do.')
        logger.info(f'{num_detectable} detectable lenses found')
        logger.info(f'{num_detectable / pipeline.config["survey"]["area"] / pipeline.runs:.2f} per square degree')
        stop = time.time()
        execution_time = util.print_execution_time(start, stop, return_string=True)
        util.write_execution_time(execution_time, SCRIPT_NAME, os.path.join(pipeline.pipeline_dir, 'execution_times.json'))
        return

    logger.info(f'Processing {len(filtered_tuple_list)} run(s) with jax_platform={jax_platform!r}')

    if jax_platform == 'gpu':
        num_detectable += _run_sequential(filtered_tuple_list, compilation_cache_dir, jax_platform)
    else:
        num_detectable += _run_spawn_pool(pipeline, filtered_tuple_list, compilation_cache_dir, jax_platform)

    logger.info(f'{num_detectable} detectable lenses found')
    logger.info(f'{num_detectable / pipeline.config["survey"]["area"] / pipeline.runs:.2f} per square degree')

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME, os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def _verify_jax_platform(jax_platform):
    """Spawn a one-shot subprocess to check ``jax.devices(jax_platform)`` works.

    Imports jax in *this* process would lock its platform for the rest of the
    run; we never want that in main, which only orchestrates workers.
    """
    code = (
        "import logging;"
        "logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL);"
        "import os, sys;"
        f"os.environ['JAX_PLATFORM_NAME']={jax_platform!r};"
        f"os.environ['JAX_PLATFORMS']={jax_platform!r};"
        "import jax;"
        f"devs=jax.devices({jax_platform!r});"
        "print(len(devs))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, check=False, timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f'Timed out probing jax platform {jax_platform!r}.') from e
    if proc.returncode != 0:
        last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '(no stderr)'
        raise RuntimeError(
            f'JAX platform {jax_platform!r} unavailable on this host: {last}'
        )
    logger.info(f"JAX platform {jax_platform!r} ready ({proc.stdout.strip()} device(s))")


def _resolve_worker_count(pipeline, count):
    """``cores.script_01b_jax`` if present, otherwise fall back to ``cores.script_01b``."""
    cores = pipeline.config['cores']
    if f'script_{SCRIPT_NAME}' in cores:
        configured = cores[f'script_{SCRIPT_NAME}']
    else:
        configured = cores['script_01b']
        logger.info(
            f"cores.script_{SCRIPT_NAME} not set; falling back to cores.script_01b={configured}"
        )
    return min(configured, count)


def _run_spawn_pool(pipeline, tuple_list, compilation_cache_dir, jax_platform):
    workers = _resolve_worker_count(pipeline, len(tuple_list))
    logger.info(f'Spawning {workers} JAX-CPU worker(s)')

    num_detectable = 0
    ctx = multiprocessing.get_context('spawn')
    # Each worker pops one CPU id and pins itself with sched_setaffinity. JAX's
    # CPU runtime allocates thread pools sized to hardware_concurrency() that
    # XLA_FLAGS does not cap; affinity is what actually keeps the run queue
    # near `workers` instead of `workers * ~3`.
    cpu_queue = ctx.Queue()
    for cpu in range(workers):
        cpu_queue.put(cpu)
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(jax_platform, compilation_cache_dir, cpu_queue),
        ) as executor:
            futures = [executor.submit(run_slsim, task) for task in tuple_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Runs'):
                num_detectable += future.result()
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    return num_detectable


def _run_sequential(tuple_list, compilation_cache_dir, jax_platform):
    """GPU path: initialize JAX in main and run runs sequentially.

    Unlike step 04 (where lenses are pickled up front and can be bucketed by
    lens_model_list), runs in step 01b draw their populations lazily, so we
    just iterate run-by-run and rely on the shared JIT cache to amortize
    compilation across runs.
    """
    _worker_init(jax_platform, compilation_cache_dir, None)

    num_detectable = 0
    for task in tqdm(tuple_list, total=len(tuple_list), desc='Runs'):
        num_detectable += run_slsim(task)
    return num_detectable


def _worker_init(jax_platform, compilation_cache_dir, cpu_queue):
    """Spawn worker entry point: pin platform, enable JIT cache, cache JAXXED set."""
    # Pin this worker to a single core (Linux only). JAX's CPU runtime sizes
    # its Eigen/dispatch pools from hardware_concurrency(), which XLA_FLAGS
    # cannot cap; sched_setaffinity is what actually bounds load to ~workers
    # instead of ~workers*3.
    if cpu_queue is not None and hasattr(os, 'sched_setaffinity'):
        try:
            cpu_id = cpu_queue.get_nowait()
            os.sched_setaffinity(0, {cpu_id})
        except Exception:
            pass

    os.environ['JAX_PLATFORM_NAME'] = jax_platform
    os.environ['JAX_PLATFORMS'] = jax_platform
    # Unconditional assignment: setdefault would silently skip if XLA_FLAGS was
    # inherited from the calling shell, leaving JAX with its default multi-
    # threaded Eigen pool. No-op effects for the GPU path.
    os.environ['XLA_FLAGS'] = ' '.join([
        '--xla_cpu_multi_thread_eigen=false',
        '--xla_force_host_platform_device_count=1',
    ])

    # JAX runs discover_pjrt_plugins() unconditionally on first import and
    # logs the cuda12 plugin's cuInit() failure at ERROR level when no GPU is
    # visible. The failure is caught internally and JAX falls back to CPU, so
    # the traceback is pure noise; silence the bridge logger before importing.
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

    import jax
    if compilation_cache_dir:
        jax.config.update('jax_compilation_cache_dir', compilation_cache_dir)
        # Defaults skip caching short compilations / small artifacts; lower
        # thresholds so every jaxtronomy kernel ends up on disk.
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)

    global _JAXXED_MODELS_SET
    from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS
    _JAXXED_MODELS_SET = set(_JAXXED_MODELS)


def _enable_jax_on_lens(lens):
    """Flip ``lens.use_jax[i]`` to True wherever the profile is JAX-supported."""
    use_jax = list(lens.use_jax) if lens.use_jax is not None else [False] * len(lens.lens_model_list)
    if len(use_jax) != len(lens.lens_model_list):
        use_jax = [False] * len(lens.lens_model_list)
    lens.use_jax = [
        True if lens.lens_model_list[i] in _JAXXED_MODELS_SET else bool(use_jax[i])
        for i in range(len(lens.lens_model_list))
    ]


def run_slsim(tuple):
    # unpack tuple
    run, detector, table_path, config, output_dir, psf_cache_dir, instrument = tuple

    # reproducible per-run seeding
    global_seed = config.get('seed', 42)
    np.random.seed(hash((global_seed, int(run))) % (2**32))

    # suppress warnings
    if config['suppress_warnings']:
        warnings.filterwarnings("ignore")

    # retrieve configuration parameters
    limit = config['limit']
    snr_config = config['snr']
    show_progress_bar = config['show_progress_bar']
    engine_params = config['imaging']['engine_params']
    survey_config = config['survey']
    area = survey_config['area']
    bands = survey_config['bands']
    cosmo = survey_config['cosmo']
    use_real_sources = survey_config['use_real_sources']
    use_slhammocks_pipeline = survey_config['use_slhammocks_pipeline']
    speed_factor = survey_config.get('speed_factor', 1)
    num_pix = config['psf']['num_pixes'][0]
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
    _snr_pos = snr_config.get('snr_detector_position', (2554, 2554))
    snr_detector_position = (_snr_pos[0], _snr_pos[1])

    # set run identifier
    if instrument.name == 'Roman':
        run_id = f'{run}_{roman_util.get_sca_string(detector).lower()}'
    elif instrument.name == 'HWO' or instrument.name == 'JWST':
        run_id = str(run).zfill(4)
    else:
        raise ValueError(f"Run identifier not implemented for {instrument.name}.")

    # load filters
    if config['pipeline_label'] == 'all':
        from mejiro.instruments.hst import HST
        from mejiro.instruments.jwst import JWST
        from mejiro.instruments.roman import Roman
        from mejiro.instruments.hwo import HWO
        speclite_filters = Roman.load_speclite_filters(detector='SCA01')
        speclite_filters = HST.load_speclite_filters()
        speclite_filters = JWST.load_speclite_filters()
        speclite_filters = HWO.load_speclite_filters()
    elif instrument.name == 'Roman':
        detector_string = roman_util.get_sca_string(detector).lower()
        filter_args = {'detector': detector_string}
        speclite_filters = instrument.load_speclite_filters(**filter_args)
    elif instrument.name == 'HWO' or instrument.name == 'JWST':
        filter_args = {}
        speclite_filters = instrument.load_speclite_filters(**filter_args)
    else:
        raise ValueError(f"Speclite filter loading not implemented for {instrument.name}.")

    logger.info(f'Loaded {instrument.name} filter response curve(s): {speclite_filters.names}')

    # load pre-computed galaxy table
    galaxy_data = util.unpickle(table_path)
    logger.info(f'Loaded galaxy table from {table_path}')

    # set survey parameters
    sky_area = Quantity(value=area, unit='deg2')

    # define cuts on the intrinsic deflector and source populations
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

    # reconstruct the lens population from pre-computed tables
    logger.info('Reconstructing galaxy population from pre-computed tables...')
    if use_slhammocks_pipeline:
        lens_galaxies = deflectors.CompoundLensHalosGalaxies(
            halo_galaxy_list=galaxy_data['halo_galaxies'],
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light={},
            cosmo=cosmo,
            sky_area=sky_area,
        )
    else:
        lens_galaxies = deflectors.AllLensGalaxies(
            red_galaxy_list=galaxy_data['red_galaxies'],
            blue_galaxy_list=galaxy_data['blue_galaxies'],
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light={},
            cosmo=cosmo,
            sky_area=sky_area,
        )
    real_galaxy_kwargs = {
        "extended_source_type": "catalog_source",
        "extended_source_kwargs": survey_config["catalog_source_kwargs"]
    } if use_real_sources else {}
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_data['blue_galaxies'],
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
    logger.info('Reconstructed galaxy population')

    # get PSF parameters for SNR calculation
    instrument_psf_kwargs = {
        'band': snr_band,
        'detector': snr_detector,
        'detector_position': snr_detector_position,
        'oversample': snr_supersampling_factor,
        'num_pix': num_pix,
        'check_cache': True,
        'psf_cache_dir': psf_cache_dir,
    }
    kwargs_psf = instrument.get_psf_kwargs(**instrument_psf_kwargs)

    # draw the total lens population
    if survey_config['total_population']:
        logger.info('Identifying lenses...')
        kwargs_lens_total_cut = {
            'min_image_separation': 0,
            'max_image_separation': 10,
            'mag_arc_limit': None
        }
        total_lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_total_cut,
                                                          speed_factor=speed_factor)
        logger.info(f'Number of total lenses: {len(total_lens_population)}')

        # compute SNRs and save
        logger.info(f'Computing SNRs for {len(total_lens_population)} lenses')
        snr_list = []
        num_exceptions = 0
        for candidate in tqdm(total_lens_population, desc=f'Run {run}: SNR candidates', disable=not show_progress_bar):
            strong_lens = GalaxyGalaxy.from_slsim(candidate, bands=bands, use_jax=False)
            _enable_jax_on_lens(strong_lens)

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
                                            pieces=True)
            exposure = Exposure(synthetic_image=synthetic_image,
                                exposure_time=snr_exposure_time,
                                engine='galsim',
                                engine_params=engine_params)
            snr, _ = snr_calculation.get_snr(exposure=exposure,
                                            snr_per_pixel_threshold=snr_per_pixel_threshold)
            snr_list.append(snr)

            if snr is None:
                num_exceptions += 1

        if len(total_lens_population) > 0:
            logger.info(f'Percentage of exceptions: {num_exceptions / len(total_lens_population) * 100:.2f}%')
        else:
            warnings.warn('No systems in total population. Consider revising the galaxy population.')
            return 0

        # save other params to CSV
        if survey_config["write_to_csv"]:
            total_pop_csv = os.path.join(output_dir, f'total_pop_{run_id}.csv')
            logger.info(f'Writing total population to {total_pop_csv}')
            slsim_util.write_lens_population_to_csv(total_pop_csv, total_lens_population, snr_list, bands=bands, show_progress_bar=False)

    # draw initial detectable lens population
    logger.info('Identifying detectable lenses...')
    kwargs_lens_detectable_cut = {
        'min_image_separation': survey_config['min_image_separation'],
        'max_image_separation': survey_config['max_image_separation'],
        'mag_arc_limit': {survey_config['mag_arc_limit_band']: survey_config['mag_arc_limit']}
    }
    lens_population = lens_pop.draw_population(kwargs_lens_cuts=kwargs_lens_detectable_cut,
                                                speed_factor=speed_factor)
    logger.info(f'Number of detectable lenses from first set of criteria: {len(lens_population)}')

    # apply additional detectability criteria
    detectable_gglenses, detectable_snr_list, masked_snr_array_list = [], [], []
    for candidate in tqdm(lens_population, desc=f'Run {run}: Detectable candidates', disable=not show_progress_bar):
        # convert from SLSim gglens to mejiro GalaxyGalaxy
        try:
            strong_lens = GalaxyGalaxy.from_slsim(candidate, bands=bands, use_jax=False)
            _enable_jax_on_lens(strong_lens)
        except Exception as e:
            logger.error(f'Could not create mejiro GalaxyGalaxy from SLSim Lens: {e}')
            continue

        # TODO do something with the substract lens flag
        # TODO do something with the add subhalos flag

        # pre-filter: magnification check (before expensive SNR calculation)
        magnification = strong_lens.physical_params['magnification']
        if magnification < survey_config['magnification']:
            continue

        # pre-filter: image position check
        image_positions = strong_lens.get_image_positions()
        if len(image_positions[0]) < 2:
            continue

        # criterion 1: SNR
        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                        instrument=instrument,
                                        band=snr_band,
                                        fov_arcsec=snr_fov_arcsec,
                                        instrument_params={'detector': snr_detector, 'detector_position': snr_detector_position},
                                        kwargs_numerics=snr_kwargs_numerics,
                                        kwargs_psf=kwargs_psf,
                                        pieces=True)
        exposure = Exposure(synthetic_image=synthetic_image,
                            exposure_time=snr_exposure_time,
                            engine='galsim',
                            engine_params=engine_params)
        snr, _ = snr_calculation.get_snr(exposure=exposure,
                                                        snr_per_pixel_threshold=snr_per_pixel_threshold)
        if snr is None or snr < snr_config['snr_threshold']:
            continue

        # if both criteria satisfied, consider detectable
        detectable_gglenses.append(candidate)
        detectable_snr_list.append(snr)
        # masked_snr_array_list.append(masked_snr_array)

        # if I've imposed a limit above this loop, exit the loop
        if limit is not None and len(detectable_gglenses) == limit:
            break

    logger.info(f'Run {run}: {len(detectable_gglenses)} detectable lens(es)')

    assert len(detectable_gglenses) == len(
        detectable_snr_list), f'Lengths of detectable_gglenses ({len(detectable_gglenses)}) and detectable_snr_list ({len(detectable_snr_list)}) do not match.'

    if len(detectable_gglenses) > 0:
        # pickle the list of slsim gglenses
        detectable_gglenses_pickle_path = os.path.join(output_dir, f'detectable_gglenses_{run_id}.pkl')
        logger.info(f'Pickling detectable gglenses to {detectable_gglenses_pickle_path}')
        util.pickle(detectable_gglenses_pickle_path, detectable_gglenses)

        # write the parameters of detectable lenses to CSV
        if survey_config["write_to_csv"]:
            detectable_pop_csv = os.path.join(output_dir, f'detectable_pop_{run_id}.csv')
            slsim_util.write_lens_population_to_csv(detectable_pop_csv, detectable_gglenses, detectable_snr_list, bands=bands, show_progress_bar=False)
    else:
        logger.info(f'No detectable lenses found for run {run}')

    # write sentinel file so a resumed execution can skip this run
    sentinel_path = os.path.join(output_dir, f'run_complete_{run_id}.txt')
    with open(sentinel_path, 'w') as f:
        f.write(str(len(detectable_gglenses)))

    return len(detectable_gglenses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate survey using pre-computed galaxy tables (JAX variant)")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False, help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    parser.add_argument('--resume', action='store_true', default=False, help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
