"""
Runs romanisim + romancal MosaicPipeline to produce L3 (co-added mosaic) cutouts using a
sub-pixel dither pattern (default SUB4).

This is the sub-pixel sibling of ``romanisim_l3_pipeline.py``. That script dithers with the
BOXGAP4_1 gap-filling pattern (~205 arcsec offsets), so systems can only be tiled into the
~one-quadrant region where all four dither footprints overlap. Sub-pixel patterns offset the
boresight by less than one pixel, so the n-fold overlap region is essentially the entire
detector and each batch carries ~4x more systems (~1600 vs ~400-441 grid slots for the
default 10-arcsec tiles). Batch cost is dominated by the per-dither romanisim simulations
plus one MosaicPipeline run and is independent of how many tiles are placed, so this reaches
the same depth (n_dithers x single-exposure time; SUB4 matches BOXGAP4_1's four exposures)
with ~4x less compute.

Trade-off: with sub-pixel dithers every exposure puts a system on nearly the same pixels, so
fixed-pattern detector effects (flat error, hot pixels) stay correlated across the stack
instead of averaging down the way they do with large dithers. Per-exposure noise still
decorrelates (each dither gets its own rng seed).

Everything else -- tiling, simulation, mosaicking, cutout extraction, output format -- is
identical to ``romanisim_l3_pipeline.py`` (the batch worker is imported from it unchanged).
Output lands in ``<data_dir>/<pipeline_label>/05_romanisim_l3_subpixel/sca##/`` so runs can
never clobber the BOXGAP outputs in ``05_romanisim_l3/``.

Usage:
    python3 romanisim_l3_pipeline_subpixel.py --config <config.yaml> [--data_dir <dir>] [--dither-pattern SUB4]
        [--resume] [--sequential]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Parent directory of pipeline output. Overrides data_dir in the config file.
    --dither-pattern: Sub-pixel dither pattern name from WfiImagingSubpixel.txt (default
        SUB4). Note the number of dither steps sets the effective depth.
    --sequential: Process systems sequentially from the start instead of randomly when a
        limit is imposed.
    --resume: Preserve existing output and skip already-completed batches (those with a
        batch_complete_*.txt sentinel). Default is to delete existing output and rebuild
        from scratch. Note the sentinels do not record the dither pattern, so only resume
        with the same --dither-pattern the original run used.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import math
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from romanisim import parameters

from mejiro.instruments.roman import Roman
from mejiro.point_wfi import PointWFI, _SUBPIXEL_FILE, _parse_dither_file
from mejiro.utils import util as mejiro_util
from mejiro.utils.pipeline_helper import PipelineHelper
from mejiro.pipeline.romanisim_pipeline import _glob_synthetic_images, _lens_id_from_pickle
# the batch worker and overlap-grid builder are pattern-agnostic (the dithered pointings
# travel in the task tuple), so reuse them unchanged
from mejiro.pipeline.romanisim_l3_pipeline import (
    POSITION_ANGLE,
    compute_overlap_skygrid,
    process_batch,
)

logger = logging.getLogger(__name__)

DEFAULT_PATTERN = 'SUB4'


def _subpixel_pattern_names():
    """Sorted names of the sub-pixel dither patterns shipped in WfiImagingSubpixel.txt."""
    return sorted(_parse_dither_file(_SUBPIXEL_FILE).keys())


def _dither_pointings(coord, pattern_name):
    """Return the dithered boresight SkyCoords of ``pattern_name`` for a target ``coord``."""
    pointings = []
    for x_off, y_off in PointWFI.get_pattern(pattern_name):
        p = PointWFI(ra=coord.ra.deg, dec=coord.dec.deg, position_angle=POSITION_ANGLE)
        p.dither(x_off, y_off)
        pointings.append(SkyCoord(ra=p.ra * u.deg, dec=p.dec * u.deg))
    return pointings


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    subpixel_patterns = _subpixel_pattern_names()
    if args.dither_pattern not in subpixel_patterns:
        raise ValueError(
            f"'{args.dither_pattern}' is not a sub-pixel dither pattern. "
            f"Available sub-pixel patterns: {subpixel_patterns}"
        )

    # read config
    with open(args.config, 'r') as f:
        import yaml
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if config['dev']:
        config['pipeline_label'] += '_dev'

    logging_level = config['logging_level']
    logging.basicConfig(
        level=getattr(logging, logging_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    # root under which this pipeline_label's step directories live; --data_dir overrides the config
    data_root = config['data_dir']
    if getattr(args, 'data_dir', None) is not None:
        logger.warning(f'Overriding data_dir in config file ({data_root}) with provided data_dir ({args.data_dir})')
        data_root = args.data_dir
    elif data_root is None:
        raise ValueError("data_dir must be specified either in the config file or via the --data_dir argument.")

    limit = config['limit']

    # each task runs one romanisim sim per dither + one MosaicPipeline (multi-GB), so default low
    num_workers = config['cores'].get('script_05_romanisim_l3', config['cores'].get('script_05_romanisim', 4))
    threads_per_worker = max(2, 64 // num_workers)
    bands = config['synthetic_image']['bands']
    seed = config['seed']
    ma_table_number = config['exposure']['ma_table_number']
    date = config['exposure']['date']
    if not isinstance(date, str):
        date = date.isoformat()
    coord = SkyCoord(ra=config['exposure']['coordinates']['ra'] * u.deg,
                     dec=config['exposure']['coordinates']['dec'] * u.deg)

    read_pattern = parameters.read_pattern[ma_table_number]
    exptime = parameters.read_time * read_pattern[-1][-1]
    logger.info(f'Total (single-exposure) exposure time (MA table {ma_table_number}): {exptime:.1f} s')

    # tile size follows the synthetic-image FOV (Roman pixel scale is band-independent)
    pixel_scale = Roman().get_pixel_scale().value
    tile_size = int(mejiro_util.set_odd_num_pix(config['synthetic_image']['fov_arcsec'], pixel_scale))

    # sub-pixel dithered boresights (shared across SCAs/bands)
    pointings = _dither_pointings(coord, args.dither_pattern)

    # discover SyntheticImage inputs (JAX variant when jaxtronomy.use_jax is set)
    synth_step = '04_jax' if config['jaxtronomy']['use_jax'] else '04'
    data_dir = os.path.join(data_root, config['pipeline_label'], synth_step)
    sca_dirs = sorted(glob(os.path.join(data_dir, 'sca*')))
    logger.info(f'Found {len(sca_dirs)} SCA directories in {data_dir}')

    pickles_by_sca_band = {}
    for sca_dir in sca_dirs:
        sca_num = int(os.path.basename(sca_dir)[3:])
        by_band = defaultdict(list)
        for p in _glob_synthetic_images(sca_dir):
            stem = os.path.basename(p).replace('.pkl', '').replace('.npz', '')
            by_band[stem.split('_')[-1]].append(p)
        pickles_by_sca_band[sca_num] = dict(by_band)

    # output directory (separate from the BOXGAP script's 05_romanisim_l3)
    output_dir = os.path.join(data_root, config['pipeline_label'], '05_romanisim_l3_subpixel')
    os.makedirs(output_dir, exist_ok=True)

    if not args.resume:
        existing = [p for p in glob(os.path.join(output_dir, '**', '*'), recursive=True)
                    if os.path.isfile(p)]
        if existing:
            logger.warning(
                f'Deleting {len(existing)} existing output file(s) in {output_dir} and '
                f'rebuilding from scratch. Pass --resume to keep them.'
            )
            mejiro_util.clear_directory(output_dir)

    logger.info(f'Dither pattern: {args.dither_pattern} ({len(pointings)} pointings, '
                f'effective depth {len(pointings)}x{exptime:.1f} s = {len(pointings) * exptime:.1f} s)')
    logger.info(f'Tile size: {tile_size}x{tile_size} ({pixel_scale * tile_size:.2f} arcsec)')
    logger.info(f'Bands: {bands}')

    # optional global cap: process only the first N systems (lowest UIDs) across all SCAs.
    # Unlike `limit` (which is applied per SCA), this selects N systems in total.
    global_allowed = None
    if args.max_systems is not None:
        all_uids = sorted({
            _lens_id_from_pickle(p, band)
            for bd in pickles_by_sca_band.values()
            for band, ps in bd.items()
            for p in ps
        })
        global_allowed = set(all_uids[:args.max_systems])
        logger.info(f'Global cap: processing the first {len(global_allowed)} of {len(all_uids)} systems')

    # build task list
    tasks = []
    for sca_num, bands_dict in sorted(pickles_by_sca_band.items()):
        sca_output_dir = os.path.join(output_dir, f'sca{str(sca_num).zfill(2)}')
        os.makedirs(sca_output_dir, exist_ok=True)

        # pick the lens-ID subsample once per SCA so every band processes the same systems
        all_lens_ids_per_band = {
            band: sorted({_lens_id_from_pickle(p, band) for p in ps})
            for band, ps in bands_dict.items()
        }
        union_lens_ids = sorted(set().union(*all_lens_ids_per_band.values())) if all_lens_ids_per_band else []
        if global_allowed is not None:
            selected_lens_ids = [u for u in union_lens_ids if u in global_allowed]
        elif limit is not None and limit < len(union_lens_ids):
            if args.sequential:
                selected_lens_ids = union_lens_ids[:limit]
            else:
                rng = np.random.default_rng(seed + sca_num)
                selected_lens_ids = sorted(rng.choice(union_lens_ids, limit, replace=False).tolist())
            logger.info(f'SCA {sca_num:02d}: limiting to {limit} lens system(s)')
        else:
            selected_lens_ids = union_lens_ids
        selected_set = set(selected_lens_ids)

        for band_idx, band in enumerate(bands):
            if band not in bands_dict:
                logger.info(f'Skipping SCA {sca_num:02d}, {band}: no pickles found')
                continue

            all_pickles = sorted(p for p in bands_dict[band] if _lens_id_from_pickle(p, band) in selected_set)
            if not all_pickles:
                continue

            # capacity = number of overlap-grid slots available for this SCA/band
            _, source_skies = compute_overlap_skygrid(pointings, sca_num, band, ma_table_number, read_pattern, date, tile_size)
            capacity = len(source_skies)
            if capacity == 0:
                logger.warning(f'SCA {sca_num:02d}, {band}: empty overlap region; skipping')
                continue

            n_batches = math.ceil(len(all_pickles) / capacity)
            logger.info(f'SCA {sca_num:02d}, {band}: {len(all_pickles)} systems, '
                        f'{capacity} slots/batch, {n_batches} batch(es)')

            for batch_idx in range(n_batches):
                batch_pickles = all_pickles[batch_idx * capacity:(batch_idx + 1) * capacity]
                tasks.append((
                    batch_pickles, sca_num, band, batch_idx, n_batches, sca_output_dir,
                    exptime, seed, ma_table_number, date, read_pattern, pointings,
                    band_idx, threads_per_worker, tile_size,
                ))

    def _batch_sentinel(sca_output_dir, sca_num, band, batch_idx):
        return os.path.join(sca_output_dir, f'batch_complete_sca{sca_num:02d}_{band}_batch{batch_idx}.txt')

    if args.resume:
        total_batches = len(tasks)
        tasks = [t for t in tasks if not os.path.exists(_batch_sentinel(t[5], t[1], t[2], t[3]))]
        logger.info(f'Resuming: {total_batches - len(tasks)} of {total_batches} batch(es) already '
                    f'complete, {len(tasks)} remaining.')

    if not tasks:
        logger.info('All batches already complete. Nothing to do.')
        stop = time.time()
        logger.info(f'Total execution time: {mejiro_util.print_execution_time(start, stop, return_string=True)}')
        return

    logger.info(f'Submitting {len(tasks)} batch(es) with {num_workers} workers')

    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for _ in tqdm(pool.imap_unordered(process_batch, tasks), total=len(tasks)):
            pass

    stop = time.time()
    logger.info(f'Total execution time: {mejiro_util.print_execution_time(start, stop, return_string=True)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run romanisim + MosaicPipeline with a sub-pixel dither pattern to produce L3 cutouts.")
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    parser.add_argument('--dither-pattern', type=str, default=DEFAULT_PATTERN,
                        help='Sub-pixel dither pattern from WfiImagingSubpixel.txt (default SUB4).')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--max-systems', type=int, default=None,
                        help='Process only the first N systems (lowest UIDs) across all SCAs, in total.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Preserve existing output and skip already-completed batches.')
    args = parser.parse_args()
    main(args)
