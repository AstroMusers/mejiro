#!/usr/bin/env python
# coding: utf-8

import datetime
import matplotlib
import os
import random
import sys
import time

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from skypy.pipeline import Pipeline
from slsim.Observations.roman_speclite import configure_roman_filters, filter_names
from pprint import pprint
from glob import glob
from tqdm import tqdm
from stips.scene_module import SceneModule
from stips.observation_module import ObservationModule
import speclite
from lenstronomy.Util import data_util
import pandas as pd
import stips
from multiprocessing import Pool
import multiprocessing


def roman_mag_to_cps(mag, band):
    if band == 'F106':
        mag_zero_point = 26.44
    elif band == 'F129':
        mag_zero_point = 26.40
    elif band == 'F184':
        mag_zero_point = 25.95
    else:
        raise ValueError('Unknown band')

    return data_util.magnitude2cps(mag, mag_zero_point)


def batch_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def main():
    repo_dir = '/grad/bwedig/mejiro'
    data_dir = '/data/bwedig/mejiro'

    # enable use of local packages
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    from mejiro.utils import util

    os.nice(19)

    start = time.time()

    print(stips.__env__report__)

    num_images = 1  # 18
    num_detectors = 1
    detector_list = [f'SCA{str(i + 1).zfill(2)}' for i in range(num_detectors)]
    bands = ['F106', 'F129', 'F184']
    # bands = ['F106']
    exposure_time = 146

    obs_prefix = 'hlwas_sim'
    obs_ra = 15.
    obs_dec = -30.

    skypy_config_path = os.path.join(repo_dir, 'paper', 'supplemental',
                                     'roman_hlwas_single_detector.yml')

    # set output directory
    output_dir = os.path.join(data_dir, 'STIPS_dev')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # compute detector side in decimal degrees
    detector_side_arcsec = 4096 * 0.11
    detector_side_deg = detector_side_arcsec / 3600
    coord_offset = detector_side_deg / 2

    # load Roman WFI filters
    configure_roman_filters()
    roman_filters = filter_names()
    roman_filters.sort()
    _ = speclite.filters.load_filters(*roman_filters[:8])

    # tuple the parameters
    tuple_list = [(j, num_detectors, detector_list, bands, exposure_time, obs_prefix, obs_ra, obs_dec, output_dir,
                   coord_offset, skypy_config_path) for j in range(num_images)]

    # split up the lenses into batches based on core count
    cpu_count = multiprocessing.cpu_count()
    process_count = num_images
    print(f'Spinning up {process_count} process(es) on {cpu_count} core(s)')

    # batch
    generator = batch_list(tuple_list, process_count)
    batches = list(generator)

    # process the batches
    for batch in tqdm(batches):
        pool = Pool(processes=process_count)
        pool.map(run_stips, batch)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')


def run_stips(tuple):
    # unpack tuple
    j, num_detectors, detector_list, bands, exposure_time, obs_prefix, obs_ra, obs_dec, output_dir, coord_offset, skypy_config_path = tuple

    # execute SkyPy galaxy simulation pipeline
    np.random.seed()
    pipeline = Pipeline.read(skypy_config_path)
    pipeline.execute()

    galaxies = []
    n_sersic_list = []

    for gal in pipeline['blue']:
        n_sersic = np.random.normal(loc=1, scale=0.25, size=1)[0]
        n_sersic_list.append(n_sersic)
        galaxies.append(gal)

    for gal in pipeline['red']:
        n_sersic = np.random.normal(loc=4, scale=1., size=1)[0]
        n_sersic_list.append(n_sersic)
        galaxies.append(gal)

    output_rows = []
    id = 0

    for gal, n_sersic in tqdm(zip(galaxies, n_sersic_list), total=len(galaxies)):
        # build information that STIPS needs
        gal_dict = {}
        gal_dict['id'] = id
        gal_dict['ra'] = np.random.uniform(low=obs_ra - coord_offset, high=obs_ra + coord_offset, size=1)[0]
        gal_dict['dec'] = np.random.uniform(low=obs_dec - coord_offset, high=obs_dec + coord_offset, size=1)[0]
        gal_dict['F106'] = roman_mag_to_cps(mag=gal['mag_F106'], band='F106')
        gal_dict['F129'] = roman_mag_to_cps(mag=gal['mag_F129'], band='F129')
        gal_dict['F184'] = roman_mag_to_cps(mag=gal['mag_F184'], band='F184')
        gal_dict['type'] = 'sersic'
        gal_dict['n'] = n_sersic
        gal_dict['re'] = gal['angular_size'] * 206265  # STIPS wants units of pixels (206265 / 0.11)
        gal_dict['phi'] = np.random.uniform(low=0, high=360, size=1)[0]
        gal_dict['ratio'] = np.random.normal(loc=0.8, scale=0.1, size=1)[0]
        gal_dict['notes'] = ''
        output_rows.append(gal_dict)
        id += 1

    # sorted_rows = sorted(output_rows, key=lambda d: d['re'], reverse=True)

    # filter out brightest galaxies
    rows_to_write = [row for row in output_rows if row['re'] < 5]  # 1e2 < row['F106'] and 

    # limit
    rows_to_write = random.sample(rows_to_write, 10)

    plt.hist([gal['re'] for gal in output_rows])
    plt.xlabel('Half-light radius [Pixels]')
    plt.savefig(os.path.join(output_dir, f'hlwas_re_hist_{j}.png'))
    plt.close()

    # from pprint import pprint
    # pprint(rows_to_write)

    # write to fits file
    df = pd.DataFrame(rows_to_write)
    table = Table.from_pandas(df)
    galaxy_cat_file = os.path.join(output_dir, f'{obs_prefix}_{j}_gals_000.fits')
    table.write(galaxy_cat_file, format='fits', overwrite=True)

    # set type header so STIPS knows how to interpret it
    with fits.open(galaxy_cat_file, mode='update') as hdul:
        hdr = hdul[1].header
        hdr['TYPE'] = 'multifilter'

    scm = SceneModule(out_prefix=f'{obs_prefix}_{j}', ra=obs_ra, dec=obs_dec, out_path=output_dir)

    stellar_parameters = {
        'n_stars': 1000,  # 1000
        'age_low': 11e12,  # 7.5e12
        'age_high': 2e12,  # 7.5e12
        'z_low': -2.,
        'z_high': -2.,
        'imf': 'salpeter',
        'alpha': -2.35,
        'binary_fraction': 0.1,
        'clustered': False,
        'distribution': 'uniform',
        'radius': 275.,  # 100.
        'radius_units': 'arcsec',
        'distance_low': 0.5,  # 10.0
        'distance_high': 10.0,  # 10.0
        'offset_ra': 0.0,
        'offset_dec': 0.0
    }
    stellar_cat_file = scm.CreatePopulation(stellar_parameters)

    print("Stellar population saved to file {}".format(stellar_cat_file))
    print("Galaxy population saved to file {}".format(galaxy_cat_file))

    offset = {
        'offset_id': 1,
        'offset_centre': False,
        'offset_ra': 0,
        'offset_dec': 0,
        'offset_pa': 0.0
    }

    residuals = {
        'residual_flat': True,
        'residual_dark': True,
        'residual_cosmic': True,
        'residual_poisson': True,
        'residual_readnoise': True
    }

    observation_parameters = {
        'instrument': 'WFI',
        'filters': bands,
        'detectors': num_detectors,
        'distortion': True,
        'background': 0.15,
        'fast_galaxy': True,
        'observations_id': j,
        'exptime': exposure_time,
        'offsets': [offset],
        'psf_xbright_limit': 17
    }

    obm = ObservationModule(observation_parameters, out_prefix=f'{obs_prefix}', ra=obs_ra, dec=obs_dec,
                            residual=residuals, out_path=output_dir, fast_galaxies=True)

    # determine number of observations
    num_obs = len(observation_parameters['filters'])

    for _ in tqdm(range(num_obs)):
        obm.nextObservation()

        # output_stellar_catalogues = obm.addCatalogue(stellar_cat_file)
        output_galaxy_catalogues = obm.addCatalogue(galaxy_cat_file, fast_galaxies=True)

        # print("Output Catalogues are {} and {}".format(output_stellar_catalogues, output_galaxy_catalogues))

        obm.addError()

        fits_file, mosaic_file, params = obm.finalize(mosaic=False)

        print("Output FITS file is {}".format(fits_file))
        print("Output Mosaic file is {}".format(mosaic_file))
        print("Observation Parameters are {}".format(params))

    # import all FITS files
    fits_files = glob(os.path.join(output_dir, f'{obs_prefix}_{j}_*.fits'))
    fits_files.sort()

    for f in fits_files:
        pprint(fits.info(f))

    for f, band in zip(fits_files, bands):
        for detector in detector_list:
            image = fits.getdata(f, extname=detector)
            np.save(os.path.join(output_dir, f'{obs_prefix}_{band}_{str(j).zfill(2)}.npy'), image)


if __name__ == '__main__':
    main()
