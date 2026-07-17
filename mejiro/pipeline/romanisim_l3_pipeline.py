"""
Runs romanisim + romancal MosaicPipeline to produce L3 (co-added mosaic) cutouts.

This is the L3 sibling of ``romanisim_pipeline.py``. Where that script tiles many
SyntheticImages into one 4088x4088 detector array, runs romanisim once to make a single
L2 exposure, and extracts cutouts, this script instead:

    1. Tiles SyntheticImages only into the region where all four BOXGAP4_1 dither pointings
       overlap on the detector (the ~one-quadrant 4-fold-overlap region). Because a single
       sky position lands in a different detector quadrant in each dither, each L2 carries
       the tiled systems in just one quadrant (a different quadrant per dither: LL/LR/UR/UL).
    2. Runs romanisim four times (once per dither) to make four L2 exposures.
    3. Co-adds the four L2s with romancal's MosaicPipeline into one L3 mosaic. Every system
       is covered by all four exposures, so the mosaic has uniform weight and an effective
       exposure time about 4x the single-exposure time.
    4. Extracts each system's cutout from the mosaic (via the mosaic WCS at the
       system's sky position) and saves it as ``Exposure_<label>_<uid>_<band>.npy`` --
       the exact same output format as ``romanisim_pipeline.py``, except the pixel values
       are the mosaic's native MJy/sr surface brightness rather than L2 DN/s.

Output lands in ``<data_dir>/<pipeline_label>/05_romanisim_l3/sca##/``.

Usage:
    python3 romanisim_l3_pipeline.py --config <config.yaml> [--data_dir <dir>] [--resume] [--sequential]

Arguments:
    --config: Path to the YAML configuration file.
    --data_dir: Parent directory of pipeline output. Overrides data_dir in the config file.
    --sequential: Process systems sequentially from the start instead of randomly when a
        limit is imposed.
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
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from glob import glob
from multiprocessing import Pool

import asdf
import galsim
import matplotlib.pyplot as plt
import numpy as np
import roman_datamodels as rdm
from astropy import table, time as astro_time, units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from tqdm import tqdm

import romanisim.bandpass
from romanisim import image, parameters, wcs

from romancal.associations import asn_from_list
from romancal.pipeline import MosaicPipeline

from mejiro.instruments.roman import Roman
from mejiro.point_wfi import PointWFI, _make_wcs
from mejiro.utils import util as mejiro_util
from mejiro.utils.pipeline_helper import PipelineHelper
# reuse the L2 helpers so cutout filenames and system selection are identical
from mejiro.pipeline.romanisim_pipeline import (
    exposure_cutout_name,
    _glob_synthetic_images,
    _lens_id_from_pickle,
    _load_pickle,
    _read_detector_position,
)

logger = logging.getLogger(__name__)

PATTERN_NAME = 'BOXGAP4_1'
PA_APER = 0.0           # deg; PA of the *observatory* Y axis (galsim PA_is_FPA=False), so the
                        # detector grid sits ~+/-60 deg from north (see docs/l3_cutout_orientation.md)
POSITION_ANGLE = 0.0    # WFI position angle passed to PointWFI
DISTORTION_GUARD = 7    # px added to tile_size for the tile-center pitch (pitch = tile_size + this)
MARGIN = 50             # extra px kept clear of each detector edge
DETECTOR_SIZE = 4088

# MosaicPipeline's FluxStep requires meta.cal_step; romanisim does not populate it.
CAL_STEP = {
    'assign_wcs':        'COMPLETE',
    'dark':              'COMPLETE',
    'dark_decay':        'COMPLETE',
    'dq_init':           'COMPLETE',
    'flat_field':        'COMPLETE',
    'flux':              'N/A',
    'linearity':         'COMPLETE',
    'outlier_detection': 'N/A',
    'photom':            'COMPLETE',
    'source_catalog':    'N/A',
    'ramp_fit':          'COMPLETE',
    'refpix':            'COMPLETE',
    'saturation':        'COMPLETE',
    'skymatch':          'N/A',
    'tweakreg':          'N/A',
    'wfi18_transient':   'N/A',
}


def _dither_pointings(coord):
    """Return the four BOXGAP4_1 dithered boresight SkyCoords for a target ``coord``."""
    pointings = []
    for x_off, y_off in PointWFI.get_pattern(PATTERN_NAME):
        p = PointWFI(ra=coord.ra.deg, dec=coord.dec.deg, position_angle=POSITION_ANGLE)
        p.dither(x_off, y_off)
        pointings.append(SkyCoord(ra=p.ra * u.deg, dec=p.dec * u.deg))
    return pointings


def compute_overlap_skygrid(pointings, sca, band, ma_table_number, read_pattern, date, tile_size):
    """Build the grid of source sky positions that land on-detector in every dither.

    Mirrors :func:`mejiro.point_wfi.find_source_position` but, instead of returning a single
    position, lays a regular ``pitch``-spaced grid of ``tile_size`` x ``tile_size`` tile
    centers across the rectangular intersection (in dither-0 pixel coordinates) of all four
    dither footprints. ``pitch`` is ``tile_size + DISTORTION_GUARD``.

    Returns a tuple ``(wcses, source_skies)`` where ``wcses`` are the four galsim WCSes and
    ``source_skies`` is a list of galsim.CelestialCoord tile-center positions (row-major).
    """
    half_tile = tile_size // 2
    pitch = tile_size + DISTORTION_GUARD

    wcses = [
        _make_wcs(pt, sca, band, ma_table_number, read_pattern, date, PA_APER)
        for pt in pointings
    ]

    center = DETECTOR_SIZE // 2
    ref_sky = wcses[0].toWorld(galsim.PositionD(center, center))
    # where the dither-0 center sky position lands in each dither (integer px offsets)
    offsets = [wcses[i].toImage(ref_sky) for i in range(len(wcses))]
    dxs = [int(round(o.x)) - center for o in offsets]
    dys = [int(round(o.y)) - center for o in offsets]

    lo = half_tile + MARGIN
    hi = DETECTOR_SIZE - half_tile - MARGIN
    x0_min = max(lo - dx for dx in dxs)
    x0_max = min(hi - dx for dx in dxs)
    y0_min = max(lo - dy for dy in dys)
    y0_max = min(hi - dy for dy in dys)

    source_skies = []
    if x0_max >= x0_min and y0_max >= y0_min:
        ys = list(range(y0_min, y0_max + 1, pitch))
        xs = list(range(x0_min, x0_max + 1, pitch))
        for gy in ys:
            for gx in xs:
                source_skies.append(wcses[0].toWorld(galsim.PositionD(gx, gy)))

    return wcses, source_skies


def _cc_to_skycoord(cc):
    """galsim.CelestialCoord -> astropy.SkyCoord (for the mosaic WCS world_to_pixel)."""
    return SkyCoord(ra=(cc.ra / galsim.degrees) * u.deg, dec=(cc.dec / galsim.degrees) * u.deg)


def _place_tile(counts, electrons, cx, cy, tile_size):
    """Place a ``tile_size`` x ``tile_size`` electron tile centered at (cx, cy) into
    ``counts``, clipping at edges."""
    half_tile = tile_size // 2
    src_y0 = max(cy - half_tile, 0)
    src_y1 = min(cy + half_tile + 1, DETECTOR_SIZE)
    src_x0 = max(cx - half_tile, 0)
    src_x1 = min(cx + half_tile + 1, DETECTOR_SIZE)
    # src_*1 can be <= src_*0 when the center is far off an edge; only place real overlap
    if src_y1 > src_y0 and src_x1 > src_x0:
        tile_y0 = src_y0 - (cy - half_tile)
        tile_y1 = tile_size - ((cy + half_tile + 1) - src_y1)
        tile_x0 = src_x0 - (cx - half_tile)
        tile_x1 = tile_size - ((cx + half_tile + 1) - src_x1)
        counts[src_y0:src_y1, src_x0:src_x1] = electrons[tile_y0:tile_y1, tile_x0:tile_x1]


def _detector_y_pa_deg(imwcs):
    """Position angle (deg east of north) of the detector +y axis at the detector center.

    Passed as the resample step's ``rotation`` so the mosaic axes are parallel to the
    detector axes; with ``rotation: None`` the mosaic comes out north-up because
    romanisim's wcsinfo metadata does not describe the galsim WCS it simulates with
    (see docs/l3_cutout_orientation.md).
    """
    center = DETECTOR_SIZE // 2
    p0 = imwcs.toWorld(galsim.PositionD(center, center))
    py = imwcs.toWorld(galsim.PositionD(center, center + 100))
    dra = (py.ra - p0.ra).rad * math.cos(p0.dec.rad)
    ddec = (py.dec - p0.dec).rad
    return math.degrees(math.atan2(dra, ddec))


def _extract_cutout(mos_data, cx, cy, tile_size):
    """Extract a ``tile_size`` x ``tile_size`` cutout centered at (cx, cy) from the mosaic,
    NaN-padding if needed."""
    half_tile = tile_size // 2
    h, w = mos_data.shape
    out = np.full((tile_size, tile_size), np.nan, dtype=mos_data.dtype)
    y0 = cy - half_tile
    x0 = cx - half_tile
    sy0, sy1 = max(y0, 0), min(y0 + tile_size, h)
    sx0, sx1 = max(x0, 0), min(x0 + tile_size, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mos_data[sy0:sy1, sx0:sx1]
    return out


def main(args):
    start = time.time()

    PipelineHelper.patch_astropy_for_mejiro_v2_pickles()  # remove after re-pickling inputs under mejiro-v3

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
    logging.getLogger('romanisim').setLevel(logging.ERROR)
    logging.getLogger('romancal').setLevel(logging.WARNING)

    # root under which this pipeline_label's step directories live; --data_dir overrides the config
    data_root = config['data_dir']
    if getattr(args, 'data_dir', None) is not None:
        logger.warning(f'Overriding data_dir in config file ({data_root}) with provided data_dir ({args.data_dir})')
        data_root = args.data_dir
    elif data_root is None:
        raise ValueError("data_dir must be specified either in the config file or via the --data_dir argument.")

    limit = config['limit']

    # each task runs 4 romanisim sims + one MosaicPipeline (multi-GB), so default low
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

    # four dithered boresights (shared across SCAs/bands)
    pointings = _dither_pointings(coord)

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

    # output directory
    output_dir = os.path.join(data_root, config['pipeline_label'], '05_romanisim_l3')
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

    logger.info(f'Dither pattern: {PATTERN_NAME} ({len(pointings)} pointings)')
    logger.info(f'Tile size: {tile_size}x{tile_size} ({pixel_scale * tile_size:.2f} arcsec), '
                f'grid pitch: {tile_size + DISTORTION_GUARD} px')
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
                logger.warning(f'SCA {sca_num:02d}, {band}: empty 4-fold overlap region; skipping')
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


def process_batch(task):
    (batch_pickles, sca_num, band, batch_idx, n_batches, sca_output_dir,
     exptime, seed, ma_table_number, date, read_pattern, pointings,
     band_idx, threads_per_worker, tile_size) = task

    try:
        n_images = len(batch_pickles)
        logger.info(f'SCA {sca_num:02d}, {band}, batch {batch_idx + 1}/{n_batches}: {n_images} images')

        batch_seed = seed + sca_num * 10000 + band_idx * 1000 + batch_idx
        rng_np = np.random.default_rng(batch_seed)

        abflux = romanisim.bandpass.get_abflux(band, sca_num)
        logger.info(f'  AB flux: {abflux:.6e} e-/s per maggy, exptime: {exptime:.6f} s')

        # rebuild the dither WCSes and overlap sky grid (deterministic; cheap)
        wcses, source_skies = compute_overlap_skygrid(pointings, sca_num, band, ma_table_number, read_pattern, date, tile_size)
        source_skies = source_skies[:n_images]  # system k -> grid slot k

        # 1. load synthetic images and convert each to electrons once
        logger.info(f'  Loading {n_images} synthetic images with {threads_per_worker} threads...')
        with ThreadPoolExecutor(max_workers=threads_per_worker) as exe:
            synth_images = list(exe.map(_load_pickle, batch_pickles))

        electrons = []
        for synth in synth_images:
            smooth = np.asarray(mejiro_util.smooth_pixels(synth.data), dtype=np.float64)
            assert smooth.shape == (tile_size, tile_size), (
                f'SyntheticImage shape {smooth.shape} does not match tile_size '
                f'{tile_size} derived from config fov_arcsec'
            )
            total_electrons = synth.get_maggies() * abflux * exptime
            electrons.append((smooth / np.sum(smooth, dtype=np.float64)) * total_electrons)

        # per-batch working directory for L2/asn/coadd artifacts
        batch_dir = os.path.join(sca_output_dir, f'_work_sca{sca_num:02d}_{band}_batch{batch_idx}')
        os.makedirs(batch_dir, exist_ok=True)

        # 2. build and simulate the four dithered L2 exposures
        l2_files = []
        for d, dithered_coord in enumerate(pointings):
            counts = np.zeros((DETECTOR_SIZE, DETECTOR_SIZE), dtype=np.float64)
            for e, sky in zip(electrons, source_skies):
                src_pix = wcses[d].toImage(sky)
                _place_tile(counts, e, int(round(src_pix.x)), int(round(src_pix.y)), tile_size)

            realized = rng_np.poisson(np.clip(counts, 0, None)).astype(np.int32)
            extra_counts = galsim.ImageI(realized)

            meta = deepcopy(parameters.default_parameters_dictionary)
            meta['instrument']['detector'] = f'WFI{sca_num:02d}'
            meta['instrument']['optical_element'] = band
            meta['exposure']['ma_table_number'] = ma_table_number
            meta['exposure']['read_pattern'] = read_pattern
            meta['exposure']['start_time'] = astro_time.Time(date)
            wcs.fill_in_parameters(meta, dithered_coord, boresight=True, pa_aper=PA_APER)

            source_catalog = table.Table({
                'ra': np.array([], dtype='f8'), 'dec': np.array([], dtype='f8'),
                'type': np.array([], dtype='U3'), 'n': np.array([], dtype='f4'),
                'half_light_radius': np.array([], dtype='f4'), 'pa': np.array([], dtype='f4'),
                'ba': np.array([], dtype='f4'), band: np.array([], dtype='f4'),
            })

            logger.info(f'  Running romanisim for dither {d}...')
            im, _ = image.simulate(
                meta, source_catalog, usecrds=False, psftype='galsim', level=2,
                rng=galsim.UniformDeviate(batch_seed + d), crparam=dict(), extra_counts=extra_counts,
            )

            # FluxStep prerequisites: inject cal_step and make photometry conversions positive
            im['meta']['cal_step'] = deepcopy(CAL_STEP)
            phot = im['meta']['photometry']
            phot['conversion_megajanskys'] = abs(float(phot['conversion_megajanskys']))
            phot['pixel_area'] = abs(float(phot['pixel_area']))

            l2_path = os.path.join(batch_dir, f'dither{d:04d}_wfi{sca_num:02d}_{band.lower()}_cal.asdf')
            af = asdf.AsdfFile()
            af.tree = {'roman': im}
            af.write_to(l2_path)
            l2_files.append(l2_path)

        # 3. co-add the four L2s into an L3 mosaic
        product_name = f'mosaic_sca{sca_num:02d}_{band}_batch{batch_idx}'
        asn_path = os.path.join(batch_dir, f'{product_name}_asn.json')
        asn = asn_from_list.asn_from_list(
            [(f, 'science') for f in l2_files], product_name=product_name,
            with_exptype=True, target='',
        )
        _, serialized = asn.dump(format='json')
        with open(asn_path, 'w') as f:
            f.write(serialized)

        logger.info(f'  Running MosaicPipeline for {product_name}...')
        MosaicPipeline.call(
            asn_path, save_results=True, output_dir=batch_dir, resample_on_skycell=False,
            steps={
                'skymatch': {'skip': True},
                'outlier_detection': {'skip': True},
                'source_catalog': {'skip': True},
                # align the mosaic axes with the detector axes so tiles stay axis-aligned
                'resample': {'pixel_scale_ratio': 1.0, 'rotation': _detector_y_pa_deg(wcses[0])},
            },
        )

        coadd_files = glob(os.path.join(batch_dir, '*_coadd.asdf'))
        if not coadd_files:
            raise FileNotFoundError(f'No *_coadd.asdf produced in {batch_dir}')
        # close the mosaic before the rmtree below: on NFS an open file cannot be
        # unlinked, which silently leaves the batch_dir behind
        with rdm.open(coadd_files[0]) as mos:
            mos_wcs = mos.meta.wcs
            mos_data = np.array(mos.data, dtype=np.float32)

        # diagnostic PNG of the mosaic
        plt.imshow(mos_data, norm=LogNorm(), origin='lower')
        plt.colorbar(label='MJy/sr')
        plt.title(f'SCA {sca_num:02d}, {band} batch {batch_idx} - L3 mosaic')
        plt.savefig(os.path.join(sca_output_dir, f'sca{sca_num:02d}_{band}_batch{batch_idx}_mosaic.png'))
        plt.close()

        # 4. extract each system's cutout from the mosaic via its sky position
        for pickle_path, sky in zip(batch_pickles, source_skies):
            mx, my = mos_wcs.world_to_pixel(_cc_to_skycoord(sky))
            cutout = _extract_cutout(mos_data, int(round(float(mx))), int(round(float(my))), tile_size)
            # the mosaic x axis is antiparallel to the detector x axis (the resample
            # rotation cannot fix the parity mismatch), so flip to the input orientation
            np.save(os.path.join(sca_output_dir, exposure_cutout_name(pickle_path)), np.fliplr(cutout))

        logger.info(f'  Saved {n_images} L3 cutouts to {sca_output_dir}')

        # tidy up the heavy intermediate L2/asn/coadd artifacts
        shutil.rmtree(batch_dir, ignore_errors=True)

        with open(os.path.join(sca_output_dir,
                               f'batch_complete_sca{sca_num:02d}_{band}_batch{batch_idx}.txt'), 'w') as f:
            f.write(str(n_images))
    except Exception:
        logger.exception(f'Batch failed for SCA {sca_num:02d}, {band}, batch {batch_idx + 1}/{n_batches}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run romanisim + MosaicPipeline to produce L3 cutouts.")
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configuration file.')
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Parent directory of pipeline output. Overrides data_dir in config file if provided.')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Process systems sequentially from the start instead of randomly when limit is imposed.')
    parser.add_argument('--max-systems', type=int, default=None,
                        help='Process only the first N systems (lowest UIDs) across all SCAs, in total.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Preserve existing output and skip already-completed batches.')
    args = parser.parse_args()
    main(args)
