#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import random
import sys
import time

import matplotlib

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

    num_images = 18  # 18
    num_detectors = 1
    detector_list = [f'SCA{str(i + 1).zfill(2)}' for i in range(num_detectors)]
    bands = ['F106', 'F129', 'F184']
    # bands = ['F106']
    exposure_time = 146

    obs_prefix = 'hlwas_sim'
    obs_ra = [14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75]
    obs_dec = [-30.5, -30.25, -30, -29.75, -29.5]
    obs_coords = []
    for _, ra in enumerate(obs_ra):
        for _, dec in enumerate(obs_dec):
            obs_coords.append((ra, dec))

    # set output directory
    output_dir = os.path.join(data_dir, 'STIPS')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # get star catalog
    star_cat_file = os.path.join(repo_dir, 'paper', 'supplemental', 'full_catalog.fits')

    # tuple the parameters
    tuple_list = [(j, num_detectors, detector_list, bands, exposure_time, obs_prefix, obs_coords[j], output_dir,
                   star_cat_file) for j in range(num_images)]

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
    j, num_detectors, detector_list, bands, exposure_time, obs_prefix, obs_coords, output_dir, star_cat_file = tuple
    obs_ra, obs_dec = obs_coords
    print(f'Processing coordinates {obs_ra}, {obs_dec}')

    np.random.seed()

    # create galaxy catalog
    galaxy_parameters = {
                     'n_gals': 1000,
                     'z_low': 0.0,
                     'z_high': 3.,  # 0.2
                     'rad_low': 0.01,
                     'rad_high': 0.2,  # 2.
                     'sb_v_low': 28.0,
                     'sb_v_high': 25.0,
                     'distribution': 'uniform',
                     'clustered': False,
                     'radius': 250.0,
                     'radius_units': 'arcsec',
                     'offset_ra': 0.0,
                     'offset_dec': 0.0,
                    }

    scm = SceneModule(out_prefix=f'{obs_prefix}_{j}', ra=obs_ra, dec=obs_dec, out_path=output_dir)
    galaxy_cat_file = scm.CreateGalaxies(galaxy_parameters)
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
        'psf_xbright_limit': 18
    }

    obm = ObservationModule(observation_parameters, out_prefix=obs_prefix, ra=obs_ra, dec=obs_dec,
                            residual=residuals, out_path=output_dir, fast_galaxies=True)

    # determine number of observations
    num_obs = len(observation_parameters['filters'])

    for _ in tqdm(range(num_obs)):
        obm.nextObservation()

        output_stellar_catalogues = obm.addCatalogue(star_cat_file)
        output_galaxy_catalogues = obm.addCatalogue(galaxy_cat_file, fast_galaxies=True)

        print("Output Catalogues are {} and {}".format(output_stellar_catalogues, output_galaxy_catalogues))

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
