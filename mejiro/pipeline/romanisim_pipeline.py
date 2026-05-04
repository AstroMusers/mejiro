"""
Runs romanisim detector simulation on tiled synthetic images.

This script tiles SyntheticImage pickles into 4088x4088 detector arrays, runs romanisim
to apply detector effects, and extracts individual cutouts. Systems are processed in
batches of 3136 (56x56 grid of 73x73 tiles) until all systems for each SCA/band are
complete. Multiprocessing is used to parallelize batch processing.

Usage:
    python3 romanisim_pipeline.py --config <config.yaml>

Arguments:
    --config: Path to the YAML configuration file.
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
from mejiro.utils import util as mejiro_util

logger = logging.getLogger(__name__)

TILE_SIZE = 73
GRID_SIDE = 56
N_TILES = GRID_SIDE * GRID_SIDE  # 3136


def _load_pickle(pickle_path):
    return mejiro_util.unpickle(pickle_path)


def main(args):
    start = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    # read config
    config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', args.config)
    with open(config_file, 'r') as f:
        import yaml
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if config['dev']:
        config['pipeline_label'] += '_dev'

    num_workers = config['cores']['script_05_romanisim']
    threads_per_worker = max(2, 64 // num_workers)
    bands = config['synthetic_image']['bands']
    seed = config['seed']
    ma_table_number = config['exposure']['ma_table_number']
    date = config['exposure']['date']
    if not isinstance(date, str):
        date = date.isoformat()
    coord = SkyCoord(ra=config['exposure']['coordinates']['ra'] * u.deg, dec=config['exposure']['coordinates']['dec'] * u.deg)

    read_pattern = parameters.read_pattern[ma_table_number]
    exptime = parameters.read_time * read_pattern[-1][-1]
    logger.info(f'Total exposure time (MA table {ma_table_number}): {exptime:.1f} s')

    # discover SCA directories and group SyntheticImage pickles by SCA and band
    data_dir = os.path.join(config['data_dir'], config['pipeline_label'], '04')
    sca_dirs = sorted(glob(os.path.join(data_dir, 'sca*')))
    logger.info(f'Found {len(sca_dirs)} SCA directories in {data_dir}')

    pickles_by_sca_band = {}
    for sca_dir in sca_dirs:
        sca_name = os.path.basename(sca_dir)
        sca_num = int(sca_name[3:])

        sca_pickles = sorted(glob(os.path.join(sca_dir, 'SyntheticImage_*.pkl')))

        by_band = defaultdict(list)
        for p in sca_pickles:
            bn = os.path.basename(p)
            parts = bn.replace('.pkl', '').split('_')
            band = parts[-1]
            by_band[band].append(p)

        pickles_by_sca_band[sca_num] = dict(by_band)
        for band, ps in sorted(by_band.items()):
            logger.debug(f'  SCA {sca_num:02d}, {band}: {len(ps)} pickles')

    # output directory
    output_dir = os.path.join(config['data_dir'], config['pipeline_label'], '05_romanisim')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f'Tile size: {TILE_SIZE}x{TILE_SIZE}')
    logger.info(f'Grid: {GRID_SIDE}x{GRID_SIDE} = {N_TILES} tiles per batch')
    logger.info(f'Bands: {bands}')

    # build task list and prepare output directories
    tasks = []
    for sca_num, bands_dict in sorted(pickles_by_sca_band.items()):
        sca_output_dir = os.path.join(output_dir, f'sca{str(sca_num).zfill(2)}')
        os.makedirs(sca_output_dir, exist_ok=True)
        mejiro_util.clear_directory(sca_output_dir)

        for band_idx, band in enumerate(bands):
            if band not in bands_dict:
                logger.info(f'Skipping SCA {sca_num:02d}, {band}: no pickles found')
                continue

            all_pickles = bands_dict[band]
            n_batches = math.ceil(len(all_pickles) / N_TILES)
            logger.info(f'SCA {sca_num:02d}, {band}: {len(all_pickles)} images in {n_batches} batch(es)')

            for batch_idx in range(n_batches):
                batch_start = batch_idx * N_TILES
                batch_end = min(batch_start + N_TILES, len(all_pickles))
                batch_pickles = all_pickles[batch_start:batch_end]

                tasks.append((
                    batch_pickles,
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
                ))

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
    (batch_pickles, sca_num, band, batch_idx, n_batches,
    sca_output_dir, exptime, seed, ma_table_number, date, coord,
    band_idx, threads_per_worker) = task

    try:
        n_images = len(batch_pickles)
        logger.info(f'SCA {sca_num:02d}, {band}, batch {batch_idx + 1}/{n_batches}: {n_images} images')

        # per-batch deterministic RNGs
        batch_seed = seed + sca_num * 10000 + band_idx * 1000 + batch_idx
        rng = galsim.UniformDeviate(batch_seed)
        rng_np = np.random.default_rng(batch_seed)

        # get AB flux
        abflux = romanisim.bandpass.get_abflux(band, sca_num)
        logger.info(f'  AB flux: {abflux:.6e} e-/s per maggy, exptime: {exptime:.6f} s')

        # 1. tile synthetic images into a 4088x4088 extra_counts array
        counts = np.zeros((4088, 4088), dtype=np.float64)

        logger.info(f'  Loading {len(batch_pickles)} pickles with {threads_per_worker} threads...')
        load_start = time.time()
        with ThreadPoolExecutor(max_workers=threads_per_worker) as pool:
            synth_images = list(pool.map(_load_pickle, batch_pickles))
        load_stop = time.time()
        logger.info(f'  Loaded {len(batch_pickles)} pickles in {mejiro_util.print_execution_time(load_start, load_stop, return_string=True)}')

        for idx, synth in enumerate(synth_images):

            # deal with negative and nan pixels
            smooth_data = np.asarray(mejiro_util.smooth_pixels(synth.data), dtype=np.float64)

            # convert the units
            synth_sum = np.sum(smooth_data, dtype=np.float64)
            maggies = synth.get_maggies()
            total_electrons = maggies * abflux * exptime
            lens_electrons = (smooth_data / synth_sum) * total_electrons

            # put it in the right place
            row = idx // GRID_SIDE
            col = idx % GRID_SIDE
            r0 = row * TILE_SIZE
            c0 = col * TILE_SIZE
            counts[r0:r0 + TILE_SIZE, c0:c0 + TILE_SIZE] = lens_electrons

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
        for idx, pickle_path in enumerate(batch_pickles):
            row = idx // GRID_SIDE
            col = idx % GRID_SIDE
            r0 = row * TILE_SIZE
            c0 = col * TILE_SIZE
            cutout = result_data[r0:r0 + TILE_SIZE, c0:c0 + TILE_SIZE]

            output_name = os.path.basename(pickle_path).replace('.pkl', '.npy').replace('SyntheticImage_', 'Exposure_')
            np.save(os.path.join(sca_output_dir, output_name), cutout)

        logger.info(f'  Saved {n_images} cutouts to {sca_output_dir}')
    except Exception:
        logger.exception(f'Batch failed for SCA {sca_num:02d}, {band}, batch {batch_idx + 1}/{n_batches}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run romanisim detector simulation on tiled synthetic images.")
    parser.add_argument('--config', type=str, required=True, help='Name of the yaml configuration file.')
    args = parser.parse_args()
    main(args)
