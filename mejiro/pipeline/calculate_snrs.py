"""
Calculates signal-to-noise ratios (SNRs) for simulated exposures.

This script computes SNR values for each lensing system processed in previous
pipeline steps and saves name-SNR pairs for downstream filtering or analysis.
It reads a YAML configuration file specifying SNR calculation parameters and
supports both sequential and parallel processing modes.

The SNR is rebuilt with a galsim Exposure at snr.snr_exposure_time, except when the input
came from _05_romanisim: there the exposure time is taken from that step's sidecars instead,
so the SNR describes the depth of the data actually shipped (an L3 co-add is n_dithers
deeper than one exposure). The config value is left alone because _01b_run_survey_simulation
uses it for the detectability cut that fixed dataset membership.

Usage:
    python3 calculate_snrs.py --config <config.yaml> [--data_dir <dir>] [--sequential] [--resume] [--prev-step <step_dir>]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Parent directory of pipeline output. Overrides data_dir in the config file.
    --sequential: Run in sequential mode instead of parallel.
    --resume: Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.
    --prev-step: Pipeline step directory holding the input exposures (default: '05_romanisim').
"""
import os

# Single-thread BLAS/OMP in every worker; must happen before numpy imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from tqdm import tqdm

import logging

from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)

PREV_SCRIPT_NAME = '05_romanisim'
SCRIPT_NAME = 'snr'
SUPPORTED_INSTRUMENTS = ['roman', 'hwo']


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    # resolve which romanisim step produced the exposures (default matches the
    # standard GalSim/romanisim step; override for variants like the sub-pixel L3 run)
    prev_script_name = getattr(args, 'prev_step', None) or PREV_SCRIPT_NAME

    # initialize PipelineHelper (we handle the default wipe ourselves so we can count + warn first)
    pipeline = PipelineHelper(args, prev_script_name, SCRIPT_NAME, SUPPORTED_INSTRUMENTS,
                              delete_existing_output=False)

    output_path = os.path.join(pipeline.output_dir, 'name_snr_pairs.pkl')

    if not args.resume:
        if os.path.exists(output_path):
            logger.warning(
                f'Deleting existing output file {os.path.basename(output_path)} and '
                f'rebuilding from scratch. Pass --resume to keep it.'
            )
            os.remove(output_path)
    elif os.path.exists(output_path):
        logger.info(f'Output already exists at {output_path}; skipping.')
        return

    # retrieve configuration parameters
    snr_config = pipeline.config['snr']

    # the step-02 lens pickles carry use_jax, so jaxtronomy gets imported in the workers
    # whether or not we ask for it; pin the platform so JAX doesn't default to the GPU
    use_jax = pipeline.config['jaxtronomy']['use_jax']
    if use_jax:
        jax_platform = pipeline.config['jaxtronomy']['jax_platform']
        _verify_jax_platform(jax_platform)

    # set input and output directories; the exposure file extension depends on which
    # step-05 variant produced them (romanisim .npy vs galsim .pkl/.npz). Only the filenames
    # matter here: _rebuild_snr re-derives the SNR from the step-02 and step-04 pickles.
    extension = PipelineHelper.exposure_extension(prev_script_name, pipeline.config['imaging']['serialization'])
    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='Exposure', suffix='', extension=extension)
    elif pipeline.instrument_name == 'hwo':
        input_pickles = pipeline.retrieve_pickles(prefix='Exposure', suffix='', extension='.pkl')
    else:
        raise ValueError(f'Unknown instrument {pipeline.instrument_name}. Supported instruments are {SUPPORTED_INSTRUMENTS}.')

    if PipelineHelper.is_romanisim_step(prev_script_name):
        # The SNR has to describe the depth of the data actually shipped, and for an L3
        # co-add that is n_dithers x the single-exposure time. snr.snr_exposure_time stays
        # untouched in the config because _01b uses it for the detectability cut that fixed
        # dataset membership; only the reported SNR follows the real exposure depth. Copied
        # so pipeline.config -- pickled into every worker below -- stays unmutated.
        snr_config = dict(snr_config)
        snr_config['snr_exposure_time'], _ = PipelineHelper.romanisim_exposure_metadata(
            pipeline.input_dir, pipeline.config['exposure']['ma_table_number'])
        logger.info(f"SNR exposure time from {prev_script_name}: "
                    f"{snr_config['snr_exposure_time']:.4f} s")

    # limit the number of systems to process, if limit imposed
    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} exposure(s)' + (f' with jax_platform={jax_platform!r}' if use_jax else ''))

    # tuple the parameters
    tuple_list = [(pipeline, snr_config, input_pickle) for input_pickle in input_pickles]

    if not use_jax:
        name_snr_pairs = _run_no_jax(pipeline, tuple_list)
    elif jax_platform == 'gpu':
        name_snr_pairs = _run_jax_gpu(pipeline, tuple_list)
    else:
        name_snr_pairs = _run_jax_cpu(pipeline, tuple_list)

    util.pickle(output_path, name_snr_pairs)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


def _verify_jax_platform(jax_platform):
    """Spawn a one-shot subprocess to check ``jax.devices(jax_platform)`` works.

    Importing jax in *this* process would lock its platform for the rest of the
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


def _drive_pool(executor, tuple_list):
    """Submit every task and drain it, tearing the pool down hard on interrupt.

    Not a ``with`` block: ``Executor.__exit__`` drains every queued task before an
    ``except`` clause could cancel them. Children must be killed explicitly since
    they survive SIGINT. See ``_04_create_synthetic_images._drive_pool``.
    """
    name_snr_pairs = []
    try:
        futures = [executor.submit(calculate_snr, task) for task in tuple_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            name_snr_pairs.append(future.result())
    except BaseException:
        logger.warning('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        for process in list(getattr(executor, '_processes', {}).values()):
            process.kill()
        raise
    else:
        executor.shutdown(wait=True)
    return name_snr_pairs


def _run_no_jax(pipeline, tuple_list):
    """Non-JAX path: fork-context process pool."""
    return _drive_pool(
        ProcessPoolExecutor(max_workers=pipeline.calculate_process_count(len(tuple_list))),
        tuple_list,
    )


def _run_jax_cpu(pipeline, tuple_list):
    """JAX CPU path: spawn-context process pool, one core-pinned worker per CPU id."""
    workers = pipeline.calculate_process_count(len(tuple_list))
    logger.info(f'Spawning {workers} JAX-CPU worker(s)')

    ctx = multiprocessing.get_context('spawn')
    # Each worker pops one CPU id and pins itself before importing jax, which is
    # what bounds jaxlib's thread pools -- they are sized from the affinity mask.
    cpu_queue = ctx.Queue()
    for cpu in range(workers):
        cpu_queue.put(cpu)
    return _drive_pool(
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_jax_worker_init,
            initargs=(pipeline.config, 'cpu', cpu_queue),
        ),
        tuple_list,
    )


def _run_jax_gpu(pipeline, tuple_list):
    """JAX GPU path: run every exposure sequentially in the main process."""
    _jax_worker_init(pipeline.config, 'gpu', None)
    logger.info(f'Running {len(tuple_list)} exposure(s) sequentially on GPU.')
    return [calculate_snr(task) for task in tqdm(tuple_list)]


def _jax_worker_init(config, jax_platform, cpu_queue):
    """JAX worker setup: pin to a core (CPU path only), then pin the platform.

    Serves both JAX paths: spawn-worker initializer on CPU (``cpu_queue`` holds
    the core ids to pin to), called directly in the main process on GPU
    (``cpu_queue`` is None, no pinning).
    """
    # Spawned workers do not inherit the parent's logging level or warning filters.
    PipelineHelper.configure_logging_and_warnings(config)

    # Spawned workers do not inherit the parent's monkeypatch, so re-apply the
    # mejiro-v2 pickle compatibility shim before any lens is unpickled.
    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    if cpu_queue is not None:
        os.sched_setaffinity(0, {cpu_queue.get()})

    os.environ['JAX_PLATFORM_NAME'] = jax_platform
    os.environ['JAX_PLATFORMS'] = jax_platform
    # Unconditional assignment: setdefault would silently skip if XLA_FLAGS was
    # inherited from the calling shell. No-op effects for the GPU path.
    os.environ['XLA_FLAGS'] = ' '.join([
        '--xla_cpu_multi_thread_eigen=false',
        '--xla_force_host_platform_device_count=1',
    ])

    # Silence the cuda plugin's harmless cuInit() traceback on GPU-less hosts.
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

    import jax  # noqa: F401  -- lock in the platform before calculate_snr() imports jaxtronomy


def calculate_snr(input):
    # unpack tuple
    (pipeline, snr_config, input_pickle) = input

    # parse pickle filename to get system name and band
    basename = os.path.basename(input_pickle)
    parts = basename.split('_')
    system_name = "_".join(parts[1:-1])
    band = parts[-1].rsplit('.', 1)[0]

    # unpack snr config
    snr_per_pixel_threshold = snr_config['snr_per_pixel_threshold']

    # load exposure
    # exposure = util.unpickle(input_pickle)
    exposure = None

    # calculate SNR
    # try:
    #     snr = exposure.get_snr(snr_per_pixel_threshold=snr_per_pixel_threshold)
    # except ValueError as e:
        # logger.warning(f'Falling back to rebuild path for {input_pickle}: {e}')
    snr = _rebuild_snr(pipeline, snr_config, input_pickle, system_name, band, exposure)

    return (f'{system_name}_{band}', snr)


def _rebuild_snr(pipeline, snr_config, input_pickle, system_name, band, original_exposure):
    # locate the StrongLens pickle (step 02 — no substructure attached) and the
    # SyntheticImage pickle (step 04 — used only for instrument_params so the
    # rebuilt PSF matches the original).
    if pipeline.instrument_name == 'roman':
        sca_dir = os.path.basename(os.path.dirname(input_pickle))
        lens_pickle_path = os.path.join(pipeline.pipeline_dir, '02', sca_dir, f'lens_{system_name}.pkl')
        synth_dir = os.path.join(pipeline.step_dir('04'), sca_dir)
    else:
        lens_pickle_path = os.path.join(pipeline.pipeline_dir, '02', f'lens_{system_name}.pkl')
        synth_dir = pipeline.step_dir('04')

    # step 04 may have written either a full pickle or a lightweight .npz
    synth_pickle_path = os.path.join(synth_dir, f'SyntheticImage_{system_name}_{band}.pkl')
    if not os.path.exists(synth_pickle_path):
        synth_pickle_path = os.path.join(synth_dir, f'SyntheticImage_{system_name}_{band}.npz')

    lens = util.unpickle(lens_pickle_path)
    old_synthetic_image = util.load_synthetic_image(synth_pickle_path)
    instrument_params = old_synthetic_image.instrument_params

    supersampling_factor = snr_config['snr_supersampling_factor']
    kwargs_numerics = {
        'supersampling_factor': supersampling_factor,
        'compute_mode': snr_config['snr_supersampling_compute_mode'],
    }

    psf_config = pipeline.config['psf']
    num_pix = psf_config['num_pixes'][0]
    get_psf_args = {
        'band': band,
        'oversample': supersampling_factor,
        'num_pix': num_pix,
        'check_cache': True,
        'psf_cache_dir': pipeline.psf_cache_dir,
        'require_cached': True,
    }
    if pipeline.instrument_name == 'roman':
        get_psf_args['detector'] = instrument_params['detector']
        get_psf_args['detector_position'] = instrument_params['detector_position']
    kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

    synthetic_image = SyntheticImage(
        strong_lens=lens,
        instrument=pipeline.instrument,
        band=band,
        fov_arcsec=snr_config['snr_fov_arcsec'],
        instrument_params=instrument_params,
        kwargs_numerics=kwargs_numerics,
        kwargs_psf=kwargs_psf,
        pieces=True,
    )

    exposure = Exposure(
        synthetic_image,
        exposure_time=snr_config['snr_exposure_time'],
        engine=pipeline.config['imaging']['engine'],
        # engine_params=original_exposure.engine_params,
        engine_params=pipeline.config['imaging']['engine_params']
    )

    return exposure.get_snr(snr_per_pixel_threshold=snr_config['snr_per_pixel_threshold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate signal-to-noise ratios.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False, help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    parser.add_argument('--prev-step', dest='prev_step', type=str, default=None,
                        help=f"Name of the pipeline step directory holding the input exposures "
                             f"(default: '{PREV_SCRIPT_NAME}').")
    args = parser.parse_args()
    main(args)
