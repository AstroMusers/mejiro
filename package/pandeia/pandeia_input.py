import datetime
import time

import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.Util import data_util
from pandeia.engine.calc_utils import build_default_calc, build_default_source
from pandeia.engine.perform_calculation import perform_calculation
from tqdm import tqdm

from package.helpers.roman_params import RomanParameters


def build_pandeia_calc(csv, array, lens, band='f106', oversample_factor=1):
    calc = build_default_calc('roman', 'wfi', 'imaging')

    # set scene size settings
    calc['configuration']['dynamic_scene'] = True
    # calc['configuration']['max_scene_size'] = 5

    # set filter
    calc['configuration']['instrument']['filter'] = band.lower()  # e.g. 'f106'

    # convert array from counts/sec to astronomical magnitude
    mag_array = _convert_cps_to_magnitude(array, band)

    # add point sources to Pandeia input ('calc')
    calc = _add_point_sources(csv, calc, mag_array, lens, band, oversample_factor)

    return calc


def get_pandeia_results(calc):
    start = time.time()

    print('Performing Pandeia calculation...')
    results = perform_calculation(calc)
    print('Pandeia calculation complete')

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))

    return results, execution_time


def get_pandeia_image(calc):
    results, execution_time = get_pandeia_results(calc)

    detector = results['2d']['detector']
    detector = np.flipud(detector)

    return detector, execution_time


def get_calculation_dict(init=True):
    return {
        'noise': {
            'crs': init,
            'dark': init,
            'excess': False,  # Roman's detectors are H4RG which do not have excess noise parameters
            'ffnoise': init,
            'readnoise': init,
            'scatter': init
        },
        'effects': {
            'saturation': init
        }
    }


def _add_point_sources(csv, calc, array, lens, band='f106', oversample_factor=1):
    i = 0
    side, _ = array.shape
    print(f'Converting {array.shape} array to point sources...')
    for row_number, row in tqdm(enumerate(array), total=side):
        for item_number, item in enumerate(row):
            if i != 0:
                calc['scene'].append(build_default_source(geometry='point', telescope='roman'))

            # set brightness
            calc['scene'][i]['spectrum']['normalization']['norm_flux'] = item
            calc['scene'][i]['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
            calc['scene'][i]['spectrum']['normalization']['norm_wave'] = _get_norm_wave(csv, band)
            calc['scene'][i]['spectrum']['normalization']['norm_waveunit'] = 'microns'
            calc['scene'][i]['spectrum']['normalization']['type'] = 'at_lambda'

            # set position
            calc['scene'][i]['position']['x_offset'] = (item_number * (1 / 9) * (
                        1 / oversample_factor)) + lens.ra_at_xy_0  # arcsec
            calc['scene'][i]['position']['y_offset'] = (row_number * (1 / 9) * (
                        1 / oversample_factor)) + lens.dec_at_xy_0  # arcsec

            i += 1
    print(f'Point source conversion complete: {i} point sources')

    return calc


def _convert_cps_to_magnitude(array, band):
    lenstronomy_roman_config = Roman(band=band.upper(), psf_type='PIXEL',
                                     survey_mode='wide_area').kwargs_single_band()  # band e.g. 'F106'
    magnitude_zero_point = lenstronomy_roman_config.get('magnitude_zero_point')

    i = 0
    side, _ = array.shape
    mag_array = np.zeros(array.shape)

    for row_number, row in tqdm(enumerate(array), total=side):
        for item_number, item in enumerate(row):
            mag_array[row_number][item_number] = data_util.cps2magnitude(item, magnitude_zero_point)
            i += 1

    return mag_array


def _get_norm_wave(csv, band):
    band = band.upper()
    roman_params = RomanParameters(csv)
    filter_center_dict = roman_params.get_filter_centers()

    return filter_center_dict[band]
