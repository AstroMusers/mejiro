"""
Compacts existing step 03 output by stripping the redundant pyHalo realization.

``_03_generate_subhalos.py`` (full serialization) pickles each subhalo'd
``StrongLens`` with its pyHalo realization object attached -- ~59 MB / ~100k
``Halo`` objects, ~76% of the pickle. ``add_realization`` already bakes the
halos into ``kwargs_lens`` / ``lens_model_list`` / ``lens_redshift_list``, which
is all step 04 reads, so the realization object is pure redundancy on disk.

This one-off, resumable, atomic pass rewrites already-written 03 pickles the way
lightweight serialization would have (see ``mejiro.analysis.lensing.strip_realization``),
WITHOUT re-running pyHalo. It is safe to run whether or not step 04 has already
consumed the 03 output (step 04 does not need the realization object).

Idempotent: a lens whose ``realization`` is ``None`` (non-subhalo copy) or is
already the lightweight sentinel string is skipped. Atomic: each file is written
to ``<path>.tmp`` and ``os.replace``-d into place, so an interrupted run never
corrupts an existing pickle.

Usage:
    python3 compact_03_realizations.py --config <config.yaml> [--data_dir <dir>] [--workers N] [--dry-run]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import yaml
from tqdm import tqdm

from mejiro.analysis import lensing
from mejiro.utils import util
from mejiro.utils.pipeline_helper import PipelineHelper

import logging
logger = logging.getLogger(__name__)


def _resolve_config_path(config):
    """Accept a path or a bare name, appending .yaml/.yml when needed (mirrors PipelineHelper)."""
    if config.endswith(('.yaml', '.yml')):
        return config
    if os.path.exists(config + '.yaml'):
        return config + '.yaml'
    if os.path.exists(config + '.yml'):
        return config + '.yml'
    raise ValueError("The configuration file must be a YAML file with extension '.yaml' or '.yml'.")


def _worker_init():
    # forkserver/spawn workers do not inherit the parent's monkeypatch; re-apply
    # before any lens is unpickled.
    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()


def _needs_compaction(realization):
    # None -> non-subhalo copy; str -> already-stripped sentinel.
    return realization is not None and not isinstance(realization, str)


def _compact_one(path):
    """Load, strip, and atomically re-pickle a single 03 lens. Returns ('skip'|'compacted', bytes_saved)."""
    before = os.path.getsize(path)
    lens = util.unpickle(path)
    if not _needs_compaction(lens.realization):
        return 'skip', 0
    lensing.strip_realization(lens)
    tmp_path = path + '.tmp'
    util.pickle(tmp_path, lens)
    os.replace(tmp_path, path)
    return 'compacted', before - os.path.getsize(path)


def _inspect_one(path):
    """Dry-run probe: report whether a file would be compacted, without writing."""
    lens = util.unpickle(path)
    return 'would_compact' if _needs_compaction(lens.realization) else 'skip'


def main(args):
    start = time.time()

    # standalone script (does not go through PipelineHelper), so configure
    # logging here to surface the progress/summary messages below.
    logging.basicConfig(level=logging.INFO)

    config_path = _resolve_config_path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_dir = args.data_dir if args.data_dir is not None else config['data_dir']
    pipeline_label = config['pipeline_label']
    if config['dev']:
        pipeline_label += '_dev'
    input_dir = os.path.join(data_dir, pipeline_label, '03')

    lens_pickles = sorted(glob(os.path.join(input_dir, 'sca*', 'lens_*.pkl')))
    logger.info(f'Found {len(lens_pickles)} lens pickle(s) in {input_dir}')
    if not lens_pickles:
        return

    # Apply the pickle-compat shim in the parent too (dry-run and result unpickling).
    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()

    if args.dry_run:
        would, skip = 0, 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as executor:
            futures = {executor.submit(_inspect_one, p): p for p in lens_pickles}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Inspecting'):
                if future.result() == 'would_compact':
                    would += 1
                else:
                    skip += 1
        logger.info(f'Dry run: {would} file(s) would be compacted, {skip} already lightweight / non-subhalo.')
        return

    compacted, skipped, bytes_saved = 0, 0, 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as executor:
            futures = {executor.submit(_compact_one, p): p for p in lens_pickles}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Compacting'):
                status, saved = future.result()
                if status == 'compacted':
                    compacted += 1
                    bytes_saved += saved
                else:
                    skipped += 1
    except KeyboardInterrupt:
        logger.info('Interrupted, shutting down workers...')
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    stop = time.time()
    reclaimed = (f'{bytes_saved / 1024**3:.1f} GB' if bytes_saved >= 1024**3
                 else f'{bytes_saved / 1024**2:.0f} MB')
    logger.info(
        f'Compacted {compacted} file(s) ({reclaimed} reclaimed), skipped {skipped}.'
    )
    util.print_execution_time(start, stop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Strip pyHalo realizations from existing step 03 pickles")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, default=None, help='Optional override for the data directory in the config.')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes (each holds ~one 60 MB realization).')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', default=False,
                        help='Report how many files would be compacted without writing anything.')
    args = parser.parse_args()
    main(args)
