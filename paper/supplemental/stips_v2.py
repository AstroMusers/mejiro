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

    num_images = 1
    num_detectors = 2
    detector_list = [f'SCA{str(i + 1).zfill(2)}' for i in range(num_detectors)]
    bands = ['F106', 'F129', 'F184']
    # bands = ['F106']
    exposure_time = 146  # 10000

    obs_prefix = 'hlwas_sim'
    obs_ra = 15.
    obs_dec = -30.

    # set output directory
    output_dir = os.path.join(data_dir, 'STIPS_dev')
    util.create_directory_if_not_exists(output_dir)
    util.clear_directory(output_dir)

    # get star catalog
    star_cat_file = os.path.join(repo_dir, 'paper', 'supplemental', 'full_catalog.fits')

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
        'observations_id': 1,
        'exptime': exposure_time,
        'offsets': [offset],
        'psf_xbright_limit': 17
    }

    # create galaxy catalog
    galaxy_parameters = {
                     'n_gals': 300,
                     'z_low': 0.0,
                     'z_high': 3.,  # 0.2
                     'rad_low': 0.01,
                     'rad_high': 2.0,  # 2.
                     'sb_v_low': 28.0,
                     'sb_v_high': 24.0,
                     'distribution': 'uniform',
                     'clustered': False,
                     'radius': 250.0,
                     'radius_units': 'arcsec',
                     'offset_ra': 0.0,
                     'offset_dec': 0.0,
                    }

    scm = SceneModule(out_prefix=obs_prefix, ra=obs_ra, dec=obs_dec, out_path=output_dir)
    galaxy_cat_file = scm.CreateGalaxies(galaxy_parameters)
    print("Galaxy population saved to file {}".format(galaxy_cat_file))

    obm = ObservationModule(observation_parameters, out_prefix=obs_prefix, ra=obs_ra, dec=obs_dec,
                            residual=residuals, out_path=output_dir, fast_galaxies=True)

    # determine number of observations
    num_obs = len(observation_parameters['filters'])

    for _ in tqdm(range(num_obs)):
        obm.nextObservation()

        _ = obm.addCatalogue(star_cat_file)
        _ = obm.addCatalogue(galaxy_cat_file, fast_galaxies=True)

        obm.addError()

        fits_file, mosaic_file, params = obm.finalize(mosaic=False)

        print("Output FITS file is {}".format(fits_file))
        print("Output Mosaic file is {}".format(mosaic_file))
        print("Observation Parameters are {}".format(params))

    # import all FITS files
    fits_files = glob(os.path.join(output_dir, f'{obs_prefix}_*.fits'))
    fits_files.sort()

    for f in fits_files:
        pprint(fits.info(f))

    for f, band in zip(fits_files, bands):
        for detector in detector_list:
            image = fits.getdata(f, extname=detector)
            np.save(os.path.join(output_dir, f'{obs_prefix}_{band}_{str(j).zfill(2)}.npy'), image)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))
    print(f'Execution time: {execution_time}')

if __name__ == '__main__':
    main()
