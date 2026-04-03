# %%
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

import json
import os
import sys
import yaml
from glob import glob
from collections import defaultdict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import table, time

import galsim
from galsim import roman
from romanisim import image, parameters, catalog, psf, util, wcs
import romanisim.bandpass

import mejiro
from mejiro.utils import util as mejiro_util

# read config
config_file = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'roman_data_challenge_rung_1.yaml')
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
if config['dev']:
    config['pipeline_label'] += '_dev'

rng = galsim.UniformDeviate(42)

# %%
# discover SCA directories and group SyntheticImage pickles by SCA and band
data_dir = os.path.join(config['data_dir'], config['pipeline_label'], '04')

sca_dirs = sorted(glob(os.path.join(data_dir, 'sca*')))
print(f'Found {len(sca_dirs)} SCA directories in {data_dir}')

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
        print(f'  SCA {sca_num:02d}, {band}: {len(ps)} pickles')

# %%
# tiling and simulation parameters
TILE_SIZE = 73
GRID_SIDE = 56
N_TILES = GRID_SIDE * GRID_SIDE  # 3136

bands = config['synthetic_image']['bands']
exposure_time = config['imaging']['exposure_time']

# output directory
output_dir = os.path.join(config['data_dir'], config['pipeline_label'], '05_romanisim')
os.makedirs(output_dir, exist_ok=True)

print(f'Tile size: {TILE_SIZE}x{TILE_SIZE}')
print(f'Grid: {GRID_SIDE}x{GRID_SIDE} = {N_TILES} tiles per band')
print(f'Full array: {GRID_SIDE * TILE_SIZE}x{GRID_SIDE * TILE_SIZE}')
print(f'Bands: {bands}')
print(f'Exposure time: {exposure_time} s')

# %%
ma_table_number = 17
date           = "2027-01-01T00:00:00"
coord = SkyCoord(ra=270.0 * u.deg, dec=66.0 * u.deg)

read_pattern = parameters.read_pattern[ma_table_number]
exptime = parameters.read_time * read_pattern[-1][-1]
print(f"Total exposure time (MA table {ma_table_number}): {exptime:.1f} s")

# %%
# for each SCA, for each band: tile, simulate with romanisim, extract cutouts
for sca_num, bands_dict in sorted(pickles_by_sca_band.items()):
    sca_output_dir = os.path.join(output_dir, f'sca{str(sca_num).zfill(2)}')
    os.makedirs(sca_output_dir, exist_ok=True)
    mejiro_util.clear_directory(sca_output_dir)

    for band in bands:
        if band not in bands_dict:
            print(f'Skipping SCA {sca_num:02d}, {band}: no pickles found')
            continue

        # get AB flux
        abflux = romanisim.bandpass.get_abflux(band, sca_num)

        band_pickles = bands_dict[band][:N_TILES]
        n_images = len(band_pickles)
        print(f'Processing SCA {sca_num:02d}, {band}: {n_images} images')
        print(f'  AB flux: {abflux:.6e} e-/s per maggy, exptime: {exptime:.6f} s')

        # 1. tile synthetic images into a 4088x4088 extra_counts array
        counts = np.zeros((4088, 4088), dtype=np.float64)
        n_bad_tiles = 0
        max_tile_value = -np.inf
        min_tile_value = np.inf

        for idx, pickle_path in enumerate(tqdm(band_pickles, desc='Loading')):
            # load synthetic image
            synth = mejiro_util.unpickle(pickle_path)

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
        plt.title(f'SCA {sca_num:02d}, {band} - Tiled Synthetic Images')
        plt.savefig(os.path.join(sca_output_dir, f'sca{sca_num:02d}_{band}_tiled.png'))
        plt.close()

        # 2. add Poisson noise
        print(counts.max(), counts.min(), np.isnan(counts).sum(), np.isinf(counts).sum())
        rng_np = np.random.default_rng(42)
        realized = rng_np.poisson(counts).astype(np.int32)
        extra_counts = galsim.ImageI(realized)  # ImageI for integer type

        # 3. set up romanisim metadata with the correct SCA
        meta = deepcopy(parameters.default_parameters_dictionary)
        meta['instrument']['detector'] = f'WFI{sca_num:02d}'
        meta['instrument']['optical_element'] = band
        meta['exposure']['ma_table_number'] = ma_table_number
        meta['exposure']['read_pattern'] = parameters.read_pattern[ma_table_number]
        meta['exposure']['start_time'] = time.Time(date, format='isot')
        wcs.fill_in_parameters(meta, coord, boresight=True)

        # 4. create source table
        catalog = table.Table({
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
        print(f'  Running romanisim for SCA {sca_num:02d}, {band}...')
        im, extras = image.simulate(
            meta, catalog,
            usecrds=False,
            psftype='galsim',
            level=2,
            rng=rng,
            crparam=dict(),
            extra_counts=extra_counts,
        )

        # save the romanisim output
        mejiro_util.pickle(os.path.join(sca_output_dir, f'im_sca{sca_num:02d}_{band}.pkl'), im)
        mejiro_util.pickle(os.path.join(sca_output_dir, f'extras_sca{sca_num:02d}_{band}.pkl'), extras)

        # 6. extract cutouts and save as .npy
        result_data = im.data
        np.save(os.path.join(sca_output_dir, f'full_array_sca{sca_num:02d}_{band}.npy'), result_data)
        for idx, pickle_path in enumerate(band_pickles):
            row = idx // GRID_SIDE
            col = idx % GRID_SIDE
            r0 = row * TILE_SIZE
            c0 = col * TILE_SIZE
            cutout = result_data[r0:r0 + TILE_SIZE, c0:c0 + TILE_SIZE]

            output_name = os.path.basename(pickle_path).replace('.pkl', '.npy').replace('SyntheticImage_', 'Exposure_')
            np.save(os.path.join(sca_output_dir, output_name), cutout)

        print(f'  Saved {n_images} cutouts to {sca_output_dir}')

# %%
# verification: compare input SyntheticImages with output cutouts
sample_sca = sorted(pickles_by_sca_band.keys())[0]
sample_band = bands[0]
sample_pickles = pickles_by_sca_band[sample_sca][sample_band][:4]

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

for i, pickle_path in enumerate(sample_pickles):
    synth = mejiro_util.unpickle(pickle_path)
    output_name = os.path.basename(pickle_path).replace('.pkl', '.npy').replace('SyntheticImage_', 'Exposure_')
    cutout = np.load(os.path.join(output_dir, f'sca{str(sample_sca).zfill(2)}', output_name))

    axes[0, i].imshow(np.log10(np.clip(synth.data, 1e-10, None)), origin='lower')
    axes[0, i].set_title('Input', fontsize=8)
    axes[0, i].axis('off')

    axes[1, i].imshow(np.log10(np.clip(cutout, 1, None)), origin='lower')
    axes[1, i].set_title('Output', fontsize=8)
    axes[1, i].axis('off')

plt.suptitle(f'SCA {sample_sca:02d}, {sample_band}: SyntheticImage vs romanisim Exposure')
plt.savefig('rung_1_romanisim_verification.png', dpi=300)
plt.close()
