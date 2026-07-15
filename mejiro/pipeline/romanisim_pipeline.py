"""
Runs romanisim detector simulation on tiled synthetic images.

This script tiles SyntheticImage pickles into 4088x4088 detector arrays, runs romanisim
to apply detector effects, and extracts individual cutouts. The tile size is derived from
config['synthetic_image']['fov_arcsec'] (via util.set_odd_num_pix at Roman's 0.11"/pix),
and the grid is built dynamically as ``grid_side = 4088 // tile_size`` tiles per side,
origin-aligned to (0,0) and leaving the right/top remainder empty. Systems are processed in
batches of ``grid_side**2`` until all systems for each SCA/band are complete.
Multiprocessing is used to parallelize batch processing.

Usage:
    python3 romanisim_pipeline.py --config <config.yaml> [--resume]

Arguments:
    --config: Path to the YAML configuration file.
    --resume: Preserve existing output and skip already-completed batches (those with a
        batch_complete_*.txt sentinel). Default is to delete existing output and rebuild
        from scratch.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import logging
import math
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from copy import deepcopy
from glob import glob

import galsim
import matplotlib.pyplot as plt
import numpy as np
from astropy import table, time as astro_time
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.colors import LogNorm
from tqdm import tqdm

import romanisim.bandpass
from romanisim import image, parameters, wcs

import mejiro
from mejiro.instruments.roman import Roman
from mejiro.utils import util as mejiro_util
from mejiro.utils.pipeline_helper import PipelineHelper

logger = logging.getLogger(__name__)

DETECTOR_SIZE = 4088  # Roman WFI SCA pixel dimension


def _load_pickle(pickle_path):
    # Despite the name, this also handles lightweight .npz files via the
    # unified loader — kept as a single entry point so the ThreadPool below
    # doesn't need to branch on extension.
    try:
        return mejiro_util.load_synthetic_image(pickle_path)
    except EOFError:
        raise EOFError(f'Corrupted SyntheticImage file: {pickle_path}')


def _lens_id_from_pickle(pickle_path, band):
    bn = os.path.basename(pickle_path)
    for ext in ('.pkl', '.npz'):
        suffix = f'_{band}{ext}'
        if bn.startswith('SyntheticImage_') and bn.endswith(suffix):
            return bn[len('SyntheticImage_'):-len(suffix)]
    raise AssertionError(bn)


def exposure_cutout_name(input_path):
    """Derive the Exposure ``.npy`` cutout filename from a SyntheticImage input path.

    Handles both full ``.pkl`` pickles and lightweight ``.npz`` inputs by stripping
    whatever extension is present before appending ``.npy`` (avoids ``.npz.npy``).
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return stem.replace('SyntheticImage_', 'Exposure_') + '.npy'


def _glob_synthetic_images(directory):
    """Return SyntheticImage files in ``directory`` regardless of extension."""
    from glob import glob as _glob
    return sorted(
        _glob(os.path.join(directory, 'SyntheticImage_*.pkl'))
        + _glob(os.path.join(directory, 'SyntheticImage_*.npz'))
    )


def _read_detector_position(pickle_path):
    sidecar = pickle_path + '.psfpos.json'
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            x, y = json.load(f)['detector_position']
        return int(x), int(y)
    obj = _load_pickle(pickle_path)
    x, y = obj.instrument_params['detector_position']
    return int(x), int(y)


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

    # read config
    config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', args.config)
    with open(config_file, 'r') as f:
        import yaml
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if config['dev']:
        config['pipeline_label'] += '_dev'

    logging_level = config.get('logging_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, logging_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    limit = config.get('limit')

    num_workers = config['cores']['script_05_romanisim']
    threads_per_worker = max(2, 64 // num_workers)
    bands = config['synthetic_image']['bands']
    seed = config['seed']

    # tile size follows the synthetic-image FOV; the grid is built to fit within the
    # detector, origin-aligned to (0,0), leaving the right/top remainder empty.
    pixel_scale = Roman().get_pixel_scale().value
    tile_size = int(mejiro_util.set_odd_num_pix(config['synthetic_image']['fov_arcsec'], pixel_scale))
    grid_side = DETECTOR_SIZE // tile_size
    n_tiles = grid_side * grid_side

    divide_up_detector = int(config['psf']['divide_up_detector'])
    assert DETECTOR_SIZE % divide_up_detector == 0, (
        f'{DETECTOR_SIZE} must be divisible by divide_up_detector ({divide_up_detector})'
    )
    assert grid_side >= divide_up_detector, (
        f'grid_side ({grid_side}) must be >= divide_up_detector ({divide_up_detector}); '
        f'reduce divide_up_detector or the synthetic_image fov_arcsec'
    )
    tps = grid_side // divide_up_detector          # tiles per bucket along one axis (floor)
    tiles_per_bucket = tps * tps
    sub_px = DETECTOR_SIZE // divide_up_detector    # detector pixels per bucket along one axis
    unused = grid_side - divide_up_detector * tps
    if unused:
        logger.warning(
            f'grid_side ({grid_side}) is not divisible by divide_up_detector '
            f'({divide_up_detector}); the last {unused} grid row(s)/col(s) will go unused.'
        )
    ma_table_number = config['exposure']['ma_table_number']
    date = config['exposure']['date']
    if not isinstance(date, str):
        date = date.isoformat()
    coord = SkyCoord(ra=config['exposure']['coordinates']['ra'] * u.deg, dec=config['exposure']['coordinates']['dec'] * u.deg)

    read_pattern = parameters.read_pattern[ma_table_number]
    exptime = parameters.read_time * read_pattern[-1][-1]
    logger.info(f'Total exposure time (MA table {ma_table_number}): {exptime:.1f} s')

    # discover SCA directories and group SyntheticImage pickles by SCA and band
    # (read from the JAX step-04 variant when jaxtronomy.use_jax is set)
    synth_step = '04_jax' if config['jaxtronomy']['use_jax'] else '04'
    data_dir = os.path.join(config['data_dir'], config['pipeline_label'], synth_step)
    sca_dirs = sorted(glob(os.path.join(data_dir, 'sca*')))
    logger.info(f'Found {len(sca_dirs)} SCA directories in {data_dir}')

    pickles_by_sca_band = {}
    for sca_dir in sca_dirs:
        sca_name = os.path.basename(sca_dir)
        sca_num = int(sca_name[3:])

        sca_pickles = _glob_synthetic_images(sca_dir)

        by_band = defaultdict(list)
        for p in sca_pickles:
            bn = os.path.basename(p)
            # strip either extension before splitting; band is the final token
            stem = bn.replace('.pkl', '').replace('.npz', '')
            band = stem.split('_')[-1]
            by_band[band].append(p)

        pickles_by_sca_band[sca_num] = dict(by_band)
        for band, ps in sorted(by_band.items()):
            logger.debug(f'  SCA {sca_num:02d}, {band}: {len(ps)} pickles')

    # output directory
    output_dir = os.path.join(config['data_dir'], config['pipeline_label'], '05_romanisim')
    os.makedirs(output_dir, exist_ok=True)

    if not args.resume:
        existing = [p for p in glob(os.path.join(output_dir, '**', '*'), recursive=True)
                    if os.path.isfile(p)]
        if existing:
            logger.warning(
                f'Deleting {len(existing)} existing output file(s) in '
                f'{output_dir} and rebuilding from scratch. Pass --resume to keep them.'
            )
            mejiro_util.clear_directory(output_dir)

    logger.info(f'Tile size: {tile_size}x{tile_size} ({pixel_scale * tile_size:.2f} arcsec)')
    logger.info(f'Grid: {grid_side}x{grid_side} = {n_tiles} tiles per batch')
    logger.info(f'PSF buckets: {divide_up_detector}x{divide_up_detector} = {divide_up_detector * divide_up_detector} '
                f'({tps}x{tps} = {tiles_per_bucket} tiles per bucket)')
    logger.info(f'Bands: {bands}')

    # build task list and prepare output directories
    tasks = []
    for sca_num, bands_dict in sorted(pickles_by_sca_band.items()):
        sca_output_dir = os.path.join(output_dir, f'sca{str(sca_num).zfill(2)}')
        os.makedirs(sca_output_dir, exist_ok=True)

        # pick the lens-ID subsample once per SCA so every band processes the same
        # systems in the same order — otherwise a per-band np.random.choice puts the
        # same lens at different tile positions in each band's tiled PNG
        all_lens_ids_per_band = {
            band: sorted({_lens_id_from_pickle(p, band) for p in ps})
            for band, ps in bands_dict.items()
        }
        union_lens_ids = sorted(set().union(*all_lens_ids_per_band.values())) if all_lens_ids_per_band else []
        if limit is not None and limit < len(union_lens_ids):
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
            count = len(all_pickles)
            logger.info(f'SCA {sca_num:02d}, {band}: processing {count} image(s)')

            # resolve detector_position for each pickle (sidecar JSON fast-path, pickle-load fallback)
            with ThreadPoolExecutor(max_workers=threads_per_worker) as exe:
                positions = list(exe.map(_read_detector_position, all_pickles))

            # group by PSF bucket; preserve input (sorted) order within each bucket
            by_bucket = defaultdict(deque)
            for pickle_path, (x, y) in zip(all_pickles, positions):
                bi = min(divide_up_detector - 1, x // sub_px)
                bj = min(divide_up_detector - 1, y // sub_px)
                by_bucket[(bi, bj)].append(pickle_path)

            n_batches = max((math.ceil(len(q) / tiles_per_bucket) for q in by_bucket.values()), default=0)
            logger.info(f'SCA {sca_num:02d}, {band}: {len(all_pickles)} images in {n_batches} batch(es); '
                        f'bucket sizes: {sorted(len(q) for q in by_bucket.values())}')

            # round-robin pop up to tiles_per_bucket per bucket per batch; each batch is a
            # list of (pickle_path, tile_r, tile_c) triples placed inside the bucket's sub-grid.
            for batch_idx in range(n_batches):
                batch_items = []
                for (bi, bj), q in by_bucket.items():
                    for k in range(min(tiles_per_bucket, len(q))):
                        pickle_path = q.popleft()
                        tile_r = bj * tps + (k // tps)
                        tile_c = bi * tps + (k % tps)
                        batch_items.append((pickle_path, tile_r, tile_c))

                tasks.append((
                    batch_items,
                    sca_num,
                    band,
                    batch_idx,
                    n_batches,
                    sca_output_dir,
                    exptime,
                    seed,
                    ma_table_number,
                    date,
                    coord,
                    band_idx,
                    threads_per_worker,
                    tile_size,
                ))

    # resume: skip batches whose completion sentinel already exists
    def _batch_sentinel(sca_output_dir, sca_num, band, batch_idx):
        return os.path.join(sca_output_dir, f'batch_complete_sca{sca_num:02d}_{band}_batch{batch_idx}.txt')

    if args.resume:
        total_batches = len(tasks)
        tasks = [t for t in tasks if not os.path.exists(_batch_sentinel(t[5], t[1], t[2], t[3]))]
        skipped = total_batches - len(tasks)
        logger.info(
            f'Resuming: {skipped} of {total_batches} batch(es) already complete, '
            f'{len(tasks)} remaining.'
        )

    if not tasks:
        logger.info('All batches already complete. Nothing to do.')
        stop = time.time()
        execution_time = mejiro_util.print_execution_time(start, stop, return_string=True)
        logger.info(f'Total execution time: {execution_time}')
        return

    logger.info(f'Submitting {len(tasks)} batch(es) with {num_workers} workers')

    # process tasks in parallel; maxtasksperchild=1 recycles each worker after one batch,
    # releasing all C-extension caches (romanisim, galsim) that gc.collect() cannot touch
    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for _ in tqdm(pool.imap_unordered(process_batch, tasks), total=len(tasks)):
            pass

    stop = time.time()
    execution_time = mejiro_util.print_execution_time(start, stop, return_string=True)
    logger.info(f'Total execution time: {execution_time}')


def process_batch(task):
    (batch_items, sca_num, band, batch_idx, n_batches,
    sca_output_dir, exptime, seed, ma_table_number, date, coord,
    band_idx, threads_per_worker, tile_size) = task

    try:
        n_images = len(batch_items)
        logger.info(f'SCA {sca_num:02d}, {band}, batch {batch_idx + 1}/{n_batches}: {n_images} images')

        # per-batch deterministic RNGs
        batch_seed = seed + sca_num * 10000 + band_idx * 1000 + batch_idx
        rng = galsim.UniformDeviate(batch_seed)
        rng_np = np.random.default_rng(batch_seed)

        # get AB flux
        abflux = romanisim.bandpass.get_abflux(band, sca_num)
        logger.info(f'  AB flux: {abflux:.6e} e-/s per maggy, exptime: {exptime:.6f} s')

        # 1. tile synthetic images into a 4088x4088 extra_counts array
        counts = np.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=np.float64)

        batch_pickles = [item[0] for item in batch_items]
        logger.info(f'  Loading {len(batch_pickles)} pickles with {threads_per_worker} threads...')
        load_start = time.time()
        with ThreadPoolExecutor(max_workers=threads_per_worker) as pool:
            synth_images = list(pool.map(_load_pickle, batch_pickles))
        load_stop = time.time()
        logger.info(f'  Loaded {len(batch_pickles)} pickles in {mejiro_util.print_execution_time(load_start, load_stop, return_string=True)}')

        for (_, tile_r, tile_c), synth in zip(batch_items, synth_images):

            # deal with negative and nan pixels
            smooth_data = np.asarray(mejiro_util.smooth_pixels(synth.data), dtype=np.float64)
            assert smooth_data.shape == (tile_size, tile_size), (
                f'SyntheticImage shape {smooth_data.shape} does not match tile_size '
                f'{tile_size} derived from config fov_arcsec'
            )

            # convert the units
            synth_sum = np.sum(smooth_data, dtype=np.float64)
            maggies = synth.get_maggies()
            total_electrons = maggies * abflux * exptime
            lens_electrons = (smooth_data / synth_sum) * total_electrons

            # place inside the PSF-bucket sub-grid assigned to this image
            r0 = tile_r * tile_size
            c0 = tile_c * tile_size
            counts[r0:r0 + tile_size, c0:c0 + tile_size] = lens_electrons

        plt.imshow(counts, norm=LogNorm(), origin='lower')
        plt.colorbar(label='Electrons')
        plt.title(f'SCA {sca_num:02d}, {band} batch {batch_idx} - Tiled Synthetic Images')
        plt.savefig(os.path.join(sca_output_dir, f'sca{sca_num:02d}_{band}_batch{batch_idx}_tiled.png'))
        plt.close()

        # 2. add Poisson noise
        realized = rng_np.poisson(counts).astype(np.int32)
        extra_counts = galsim.ImageI(realized)

        # 3. set up romanisim metadata with the correct SCA
        meta = deepcopy(parameters.default_parameters_dictionary)
        meta['instrument']['detector'] = f'WFI{sca_num:02d}'
        meta['instrument']['optical_element'] = band
        meta['exposure']['ma_table_number'] = ma_table_number
        meta['exposure']['read_pattern'] = parameters.read_pattern[ma_table_number]
        meta['exposure']['start_time'] = astro_time.Time(date)
        wcs.fill_in_parameters(meta, coord, boresight=True)

        # 4. create empty source table
        source_catalog = table.Table({
            'ra': np.array([], dtype='f8'),
            'dec': np.array([], dtype='f8'),
            'type': np.array([], dtype='U3'),
            'n': np.array([], dtype='f4'),
            'half_light_radius': np.array([], dtype='f4'),
            'pa': np.array([], dtype='f4'),
            'ba': np.array([], dtype='f4'),
            band: np.array([], dtype='f4'),
        })

        # 5. simulate
        logger.info(f'  Running romanisim for SCA {sca_num:02d}, {band} batch {batch_idx}...')
        im, extras = image.simulate(
            meta, source_catalog,
            usecrds=False,
            psftype='galsim',
            level=2,
            rng=rng,
            crparam=dict(),
            extra_counts=extra_counts,
        )

        # save the romanisim output
        mejiro_util.pickle(os.path.join(sca_output_dir, f'im_sca{sca_num:02d}_{band}_batch{batch_idx}.pkl'), im)
        mejiro_util.pickle(os.path.join(sca_output_dir, f'extras_sca{sca_num:02d}_{band}_batch{batch_idx}.pkl'), extras)

        # 6. extract cutouts and save as .npy
        result_data = im.data
        np.save(os.path.join(sca_output_dir, f'full_array_sca{sca_num:02d}_{band}_batch{batch_idx}.npy'), result_data)
        for pickle_path, tile_r, tile_c in batch_items:
            r0 = tile_r * tile_size
            c0 = tile_c * tile_size
            cutout = result_data[r0:r0 + tile_size, c0:c0 + tile_size]

            output_name = exposure_cutout_name(pickle_path)
            np.save(os.path.join(sca_output_dir, output_name), cutout)

        logger.info(f'  Saved {n_images} cutouts to {sca_output_dir}')

        # write the completion sentinel only after all artifacts for this batch are on disk
        sentinel_path = os.path.join(sca_output_dir, f'batch_complete_sca{sca_num:02d}_{band}_batch{batch_idx}.txt')
        with open(sentinel_path, 'w') as f:
            f.write(str(n_images))
    except Exception:
        logger.exception(f'Batch failed for SCA {sca_num:02d}, {band}, batch {batch_idx + 1}/{n_batches}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run romanisim detector simulation on tiled synthetic images.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--resume', action='store_true', default=False, help='Preserve existing output and skip already-completed items. Default is to delete and rebuild from scratch.')
    args = parser.parse_args()
    main(args)
