"""
JAX-accelerated variant of ``_04_create_synthetic_images.py``.

Routes lenstronomy ray-shooting through jaxtronomy by flipping
``StrongLens.use_jax`` to True on every profile in ``_JAXXED_MODELS`` before
constructing the SyntheticImage.

Execution topology is selected from the YAML config's ``jaxtronomy.jax_platform``:

* ``cpu`` -- ``ProcessPoolExecutor`` with a **spawn** multiprocessing context.
  Each worker imports JAX after ``JAX_PLATFORM_NAME`` is set, and all workers
  share an on-disk JAX compilation cache so the JIT cost for each unique
  ``lens_model_list`` signature is paid only once across the whole run.

* ``gpu`` -- single-process sequential execution, bucketed by lens model
  signature so the JIT cache is reused within each bucket. One in-flight kernel
  per GPU; oversubscribing N processes onto one device thrashes memory.

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

Outputs land in ``<pipeline_dir>/04_jax/`` with the same naming convention as
step 04, so step 05 can consume either step 04 or step 04_jax outputs.

Usage:
    python3 _04_jax_create_synthetic_images.py --config <config.yaml> [--resume] [--sequential]

Config additions (optional):
    cores.script_04_jax: <int>   # falls back to cores.script_04 if absent
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
import hashlib
import json
import multiprocessing
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
from tqdm import tqdm

from mejiro.utils import roman_util, util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


PREV_SCRIPT_NAME = '03'
SCRIPT_NAME = '04_jax'
SUPPORTED_INSTRUMENTS = ['roman', 'jwst', 'hwo']

# Populated per-worker by ``_worker_init`` so create_synthetic_image can flip
# use_jax without re-importing jaxtronomy for every lens.
_JAXXED_MODELS_SET = None


def _is_deflector_only(name, seed, fraction):
    """Deterministically decide whether a system is rendered deflector-only.

    Keyed on ``seed`` + lens ``name`` (not on list order), so the same system is
    galaxy-only in every band and the choice is stable across ``--resume`` and
    the size-sort/limit reordering step 04 does. The realized fraction is
    approximate (binomial); see ``_03_generate_subhalos.py`` for an exact-count
    index-mask alternative if that is ever required.
    """
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    h = hashlib.md5(f'{seed}_deflector_only_{name}'.encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000 < fraction


def main(args):
    start = time.time()

    pipeline = PipelineHelper(args, PREV_SCRIPT_NAME, SCRIPT_NAME, SUPPORTED_INSTRUMENTS,
                              delete_existing_output=False)

    if not args.resume:
        existing = [p for p in glob(os.path.join(pipeline.output_dir, '**', '*'), recursive=True)
                    if os.path.isfile(p)]
        if existing:
            logger.warning(
                f'Deleting {len(existing)} existing output file(s) in '
                f'{pipeline.output_dir} and rebuilding from scratch. Pass --resume to keep them.'
            )
            util.clear_directory(pipeline.output_dir)

    synthetic_image_config = pipeline.config['synthetic_image']
    psf_config = pipeline.config['psf']
    jax_platform = pipeline.config.get('jaxtronomy', {}).get('jax_platform', 'cpu')

    # 'full' pickles the entire SyntheticImage (current default); 'lightweight'
    # writes a compact .npz used by the romanisim path only.
    serialization = synthetic_image_config.get('serialization', 'full')
    if serialization not in ('full', 'lightweight'):
        raise ValueError(
            f"synthetic_image.serialization must be 'full' or 'lightweight', got {serialization!r}"
        )
    output_ext = '.npz' if serialization == 'lightweight' else '.pkl'

    # Verify JAX is importable and has devices for the requested platform. Run
    # in a subprocess so the main process's JAX platform stays unlocked; we
    # never import jax in this module.
    _verify_jax_platform(jax_platform)

    # Co-locate the persistent JIT cache with the outputs so it survives across
    # runs and is shared between all spawn workers.
    compilation_cache_dir = os.path.join(pipeline.output_dir, '.jax_cache')
    os.makedirs(compilation_cache_dir, exist_ok=True)

    if pipeline.instrument_name == 'roman':
        input_pickles = pipeline.retrieve_roman_pickles(prefix='lens', suffix='', extension='.pkl')
        pipeline.create_roman_sca_output_directories()
    elif pipeline.instrument_name in ('hwo', 'jwst'):
        input_pickles = pipeline.retrieve_pickles(prefix='lens', suffix='', extension='.pkl')
    else:
        raise ValueError(
            f'Unknown instrument {pipeline.instrument_name}. '
            f'Supported instruments are {SUPPORTED_INSTRUMENTS}.'
        )

    bands = synthetic_image_config['bands']

    def _output_dir_for(input_pickle):
        if pipeline.instrument_name == 'roman':
            sca = pipeline.parse_sca_from_filename(input_pickle)
            return os.path.join(pipeline.output_dir, roman_util.get_sca_string(sca).lower())
        return pipeline.output_dir

    def _lens_name(input_pickle):
        base = os.path.basename(input_pickle)
        assert base.startswith('lens_') and base.endswith('.pkl'), base
        return base[len('lens_'):-len('.pkl')]

    def _is_complete(input_pickle):
        out_dir = _output_dir_for(input_pickle)
        name = _lens_name(input_pickle)
        for band in bands:
            if not os.path.exists(os.path.join(out_dir, f'SyntheticImage_{name}_{band}{output_ext}')):
                return False
        return True

    if args.resume:
        total_before = len(input_pickles)
        input_pickles = [p for p in input_pickles if not _is_complete(p)]
        skipped = total_before - len(input_pickles)
        logger.info(
            f'Resuming: {skipped} of {total_before} lens(es) already complete, '
            f'{len(input_pickles)} remaining.'
        )

    count = len(input_pickles)
    if pipeline.limit is not None and pipeline.limit < count:
        logger.info(f'Limiting to {pipeline.limit} lens(es)')
        if args.sequential:
            input_pickles = input_pickles[:pipeline.limit]
        else:
            input_pickles = list(np.random.choice(input_pickles, pipeline.limit, replace=False))
        count = pipeline.limit
    logger.info(f'Processing {count} lens(es) with jax_platform={jax_platform!r}')

    if count == 0:
        logger.info('Nothing to do; exiting.')
        return

    if not args.sequential:
        # Substructured lens pickles are much larger on disk; submit them first
        # to flatten the long tail.
        input_pickles.sort(key=os.path.getsize, reverse=True)

    seed = pipeline.config['seed']
    deflector_only_fraction = synthetic_image_config.get('deflector_only_fraction', 0.0)
    if deflector_only_fraction:
        logger.info(
            f'Rendering ~{deflector_only_fraction * 100:.0f}% of systems as deflector-only '
            f'(lens galaxy light only, no source or lensing)'
        )
    tuple_list = [(pipeline, synthetic_image_config, psf_config, input_pickle, serialization,
                   _is_deflector_only(_lens_name(input_pickle), seed, deflector_only_fraction))
                  for input_pickle in input_pickles]

    if jax_platform == 'gpu':
        _run_sequential_bucketed(tuple_list, compilation_cache_dir, jax_platform)
    else:
        _run_spawn_pool(pipeline, tuple_list, compilation_cache_dir, jax_platform)

    stop = time.time()
    execution_time = util.print_execution_time(start, stop, return_string=True)
    util.write_execution_time(execution_time, SCRIPT_NAME,
                              os.path.join(pipeline.pipeline_dir, 'execution_times.json'))


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
    """``cores.script_04_jax`` if present, otherwise fall back to ``cores.script_04``."""
    cores = pipeline.config['cores']
    if f'script_{SCRIPT_NAME}' in cores:
        configured = cores[f'script_{SCRIPT_NAME}']
    else:
        configured = cores['script_04']
        logger.info(
            f"cores.script_{SCRIPT_NAME} not set; falling back to cores.script_04={configured}"
        )
    return min(configured, count)


def _run_spawn_pool(pipeline, tuple_list, compilation_cache_dir, jax_platform):
    workers = _resolve_worker_count(pipeline, len(tuple_list))
    logger.info(f'Spawning {workers} JAX-CPU worker(s)')

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
            futures = {executor.submit(create_synthetic_image, task): task for task in tuple_list}
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def _run_sequential_bucketed(tuple_list, compilation_cache_dir, jax_platform):
    """GPU path: bucket by lens_model_list signature, run in main process.

    All lenses sharing a signature reuse the same JIT-compiled deflection
    kernel, so unpickling the lens (cheap relative to compilation) up front to
    derive the signature is worth it.
    """
    _worker_init(jax_platform, compilation_cache_dir, None)

    buckets = defaultdict(list)
    for task in tqdm(tuple_list, desc='Bucketing by lens_model_list'):
        input_pickle = task[3]
        lens = util.unpickle(input_pickle)
        buckets[tuple(lens.lens_model_list)].append(task)

    logger.info(
        f'Bucketed {len(tuple_list)} system(s) into {len(buckets)} '
        f'lens_model_list signature group(s); running sequentially on GPU.'
    )

    with tqdm(total=len(tuple_list)) as pbar:
        for tasks in buckets.values():
            for task in tasks:
                create_synthetic_image(task)
                pbar.update(1)


def _worker_init(jax_platform, compilation_cache_dir, cpu_queue):
    """Spawn worker entry point: pin platform, enable JIT cache, cache JAXXED set."""
    # Spawned workers do not inherit the parent's sys.modules/monkeypatch, so
    # re-apply the mejiro-v2 pickle compatibility shim here before any lens is
    # unpickled. The GPU path calls _worker_init directly in the main process.
    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    # Pin this worker to a single core (Linux only). JAX's CPU runtime sizes
    # its Eigen/dispatch pools from hardware_concurrency(), which XLA_FLAGS
    # cannot cap; sched_setaffinity is what actually bounds load to ~workers
    # instead of ~workers*3.
    if cpu_queue is not None and hasattr(os, 'sched_setaffinity'):
        try:
            # Blocking get: exactly `workers` ids are enqueued before the pool
            # is created, so this cannot deadlock or starve a later worker. A
            # non-blocking get_nowait() races the Queue's feeder thread and
            # raises Empty under host load, silently leaving the worker unpinned
            # and letting JAX's CPU pools oversubscribe cores.
            cpu_id = cpu_queue.get(timeout=30)
            os.sched_setaffinity(0, {cpu_id})
        except Exception as e:
            logger.warning(
                f'CPU affinity pin failed ({e!r}); worker runs unpinned and JAX '
                f'may oversubscribe cores.'
            )

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


def create_synthetic_image(input):
    # Import SyntheticImage lazily so the main process never pulls in
    # lenstronomy's heavy import graph; only workers need it.
    from mejiro.synthetic_image import SyntheticImage

    pipeline, synthetic_image_config, psf_config, input_pickle, serialization, deflector_only = input
    output_ext = '.npz' if serialization == 'lightweight' else '.pkl'

    bands = synthetic_image_config['bands']
    fov_arcsec = synthetic_image_config['fov_arcsec']
    supersampling_compute_mode = synthetic_image_config['supersampling_compute_mode']
    supersampling_factor = synthetic_image_config['supersampling_factor']
    pieces = synthetic_image_config['pieces']
    num_pix = psf_config['num_pixes'][0]
    divide_up_detector = psf_config.get('divide_up_detector')

    lens = util.unpickle(input_pickle)
    _enable_jax_on_lens(lens)

    kwargs_numerics = {
        "supersampling_factor": supersampling_factor,
        "compute_mode": supersampling_compute_mode,
    }

    get_psf_args = {}
    instrument_params = {}

    if pipeline.instrument_name == 'roman':
        possible_detector_positions = roman_util.divide_up_sca(divide_up_detector)
        detector_position = random.choice(possible_detector_positions)
        instrument_params['detector'] = pipeline.parse_sca_from_filename(input_pickle)
        instrument_params['detector_position'] = detector_position
        get_psf_args |= instrument_params

        sca_string = roman_util.get_sca_string(instrument_params['detector']).lower()
        output_dir = os.path.join(pipeline.output_dir, sca_string)
    else:
        output_dir = pipeline.output_dir

    for band in bands:
        output_path = os.path.join(output_dir, f'SyntheticImage_{lens.name}_{band}{output_ext}')
        failed_path = os.path.join(output_dir, f'failed_{lens.name}_{band}.pkl')
        if os.path.exists(output_path):
            continue
        if os.path.exists(failed_path):
            try:
                os.remove(failed_path)
            except OSError:
                pass

        get_psf_args |= {
            'band': band,
            'oversample': supersampling_factor,
            'num_pix': num_pix,
            'check_cache': True,
            'psf_cache_dir': pipeline.psf_cache_dir,
            'require_cached': True,
        }
        kwargs_psf = pipeline.instrument.get_psf_kwargs(**get_psf_args)

        try:
            synthetic_image = SyntheticImage(strong_lens=lens,
                                             instrument=pipeline.instrument,
                                             band=band,
                                             fov_arcsec=fov_arcsec,
                                             instrument_params=instrument_params,
                                             kwargs_numerics=kwargs_numerics,
                                             kwargs_psf=kwargs_psf,
                                             pieces=pieces,
                                             deflector_only=deflector_only)
            if serialization == 'lightweight':
                synthetic_image.save_lightweight(output_path)
            else:
                util.pickle(output_path, synthetic_image)
        except Exception as e:
            util.pickle(failed_path, lens)
            logger.warning(
                f'Error creating synthetic image for lens {lens.name} in band {band}: {e}. '
                f'Pickling to {failed_path}'
            )
            return

        if pipeline.instrument_name == 'roman':
            sidecar_path = output_path + '.psfpos.json'
            tmp_path = sidecar_path + '.tmp'
            x, y = instrument_params['detector_position']
            with open(tmp_path, 'w') as f:
                json.dump({
                    'detector_position': [int(x), int(y)],
                    'divide_up_detector': int(divide_up_detector),
                }, f)
            os.replace(tmp_path, sidecar_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic images using JAX/jaxtronomy")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
