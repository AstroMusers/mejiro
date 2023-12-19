import datetime
import os
import time

import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.Util import data_util
from pandeia.engine.calc_utils import build_default_calc, build_default_source
from pandeia.engine.perform_calculation import perform_calculation
from tqdm import tqdm

from mejiro.helpers.roman_params import RomanParameters


def build_pandeia_calc(array, lens, band='f106', max_scene_size=5, num_samples=None, oversample_factor=None,
                       suppress_output=False):
    calc = build_default_calc('roman', 'wfi', 'imaging')

    # set scene size settings
    calc['configuration']['max_scene_size'] = max_scene_size

    # set instrument
    calc['configuration']['instrument']['filter'] = band.lower()  # e.g. 'f106'

    # set detector
    calc['configuration']['detector']['ma_table_name'] = 'hlwas_imaging'

    # turn on noise sources
    calc['calculation'] = get_calculation_dict(init=True)

    # convert array from counts/sec to astronomical magnitude
    mag_array = _get_mag_array(lens, array, num_samples, band, suppress_output)

    # add point sources to Pandeia input
    norm_wave = _get_norm_wave(band)
    if num_samples:
        calc, num_point_sources = _phonion_sample(calc, mag_array, lens, num_samples, norm_wave, suppress_output)
    elif oversample_factor:
        calc, num_point_sources = _phonion_grid(calc, mag_array, lens, oversample_factor, norm_wave, suppress_output)
    else:
        raise Exception('Either provide num_samples to use sampling method or oversample_factor to use grid method')

    if not suppress_output:
        print(f'Estimated calculation time: {estimate_calculation_time(num_point_sources)}')

    return calc, num_point_sources


def get_pandeia_results(calc, suppress_output=False):
    start = time.time()

    if not suppress_output:
        print('Performing Pandeia calculation...')
    results = perform_calculation(calc)

    stop = time.time()
    execution_time = str(datetime.timedelta(seconds=round(stop - start)))

    if not suppress_output:
        print(f'Pandeia calculation complete in {execution_time}')

    return results, execution_time


def estimate_calculation_time(num_point_sources):
    seconds = round(0.0785 * num_point_sources)
    
    return str(datetime.timedelta(seconds=seconds))


def get_pandeia_image(calc, suppress_output=False):
    results, execution_time = get_pandeia_results(calc, suppress_output)

    return results['2d']['detector'], execution_time


def get_calculation_dict(init=True):
    return {
        'noise': {
            'crs': init,
            'dark': init,
            'excess': False,  # Roman's detectors are H4RG which do not have excess noise parameters
            'ffnoise': init,
            'readnoise': init,
            'scatter': False  # doesn't seem to have an effect
        },
        'effects': {
            'saturation': False  # doesn't seem to have an effect
        }
    }


def _phonion_sample(calc, mag_array, lens, num_samples, norm_wave, suppress_output=False):
    i = 0

    # loop over non-zero pixels, i.e. ignore pixels with no phonions
    for x, y in tqdm(np.argwhere(mag_array != np.inf), disable=suppress_output):
        if i != 0:
            calc['scene'].append(build_default_source(geometry='point', telescope='roman'))

        # set brightness
        calc['scene'][i]['spectrum']['normalization']['norm_flux'] = mag_array[x][y]
        calc['scene'][i]['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
        calc['scene'][i]['spectrum']['normalization']['norm_wave'] = norm_wave
        calc['scene'][i]['spectrum']['normalization']['norm_waveunit'] = 'microns'
        calc['scene'][i]['spectrum']['normalization']['type'] = 'at_lambda'

        # calculate position. NB this returns the center of the pixel
        ra, dec = lens.pixel_grid.map_pix2coord(x=x, y=y)

        # set position
        calc['scene'][i]['position']['x_offset'] = dec
        calc['scene'][i]['position']['y_offset'] = -ra  # TODO why does this make sense? it works, but why?

        i += 1

    if not suppress_output:
        print(f'Point source conversion complete: placed {i} point sources')

    return calc, i


def _get_mag_array(lens, array, num_samples, band, suppress_output):
    # normalize the image to convert it into a PDF
    sum = np.sum(array)
    normalized_array = array / sum

    # calculate flux in counts/sec of source and lens light
    source_flux_cps = lens.source_model_class.total_flux(lens.kwargs_source_amp)[0]
    lens_flux_cps = lens.lens_light_model_class.total_flux(lens.kwargs_lens_light_amp)[0]

    # get total flux in counts/sec so we know how bright to make each pixel (in counts/sec)
    total_flux_cps = source_flux_cps + lens_flux_cps
    counts_per_pixel = total_flux_cps / num_samples

    # turn array into probability distribution function and sample from it
    flattened = normalized_array.flatten()
    sample_indices = np.random.choice(a=flattened.size, p=flattened, size=num_samples)
    adjusted_indices = np.unravel_index(sample_indices, normalized_array.shape)
    adjusted_indices = np.array(list(zip(*adjusted_indices)))

    # build the sampled array
    reconstructed_array = np.zeros(array.shape)
    for x, y in adjusted_indices:
        reconstructed_array[x][y] += counts_per_pixel

    # convert from counts/sec to astronomical magnitude
    return _convert_cps_to_magnitude(reconstructed_array, band, suppress_output)


def _convert_cps_to_magnitude(array, band, suppress_output):
    lenstronomy_roman_config = Roman(band=band.upper(), psf_type='PIXEL',
                                     survey_mode='wide_area').kwargs_single_band()  # band e.g. 'F106'
    magnitude_zero_point = lenstronomy_roman_config.get('magnitude_zero_point')

    i = 0
    side, _ = array.shape
    mag_array = np.zeros(array.shape)

    for row_number, row in tqdm(enumerate(array), total=side, disable=suppress_output):
        for item_number, item in enumerate(row):
            mag_array[row_number][item_number] = data_util.cps2magnitude(item, magnitude_zero_point)
            i += 1

    return mag_array


def _convert_magnitude_to_cps(array, band, suppress_output):
    lenstronomy_roman_config = Roman(band=band.upper(), psf_type='PIXEL',
                                     survey_mode='wide_area').kwargs_single_band()  # band e.g. 'F106'
    magnitude_zero_point = lenstronomy_roman_config.get('magnitude_zero_point')

    i = 0
    side, _ = array.shape
    cps_array = np.zeros(array.shape)

    for row_number, row in tqdm(enumerate(array), total=side, disable=suppress_output):
        for item_number, item in enumerate(row):
            cps_array[row_number][item_number] = data_util.magnitude2cps(item, magnitude_zero_point)
            i += 1

    return cps_array


def _get_norm_wave(band):
    band = band.upper()
    roman_params = _get_roman_params()
    filter_center_dict = roman_params.get_filter_centers()

    return filter_center_dict[band]


def _get_roman_params():
    import mejiro
    module_path = os.path.dirname(mejiro.__file__)
    csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
    return RomanParameters(csv_path)


def _phonion_grid(calc, mag_array, lens, oversample_factor, norm_wave, suppress_output=False):
    i = 0
    side, _ = mag_array.shape

    if not suppress_output:
        print(f'Converting {mag_array.shape} array to point sources...')

    for row_number, row in tqdm(enumerate(mag_array), total=side, disable=suppress_output):
        for item_number, item in enumerate(row):
            if i != 0:
                calc['scene'].append(build_default_source(geometry='point', telescope='roman'))

            # set brightness
            calc['scene'][i]['spectrum']['normalization']['norm_flux'] = item
            calc['scene'][i]['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
            calc['scene'][i]['spectrum']['normalization']['norm_wave'] = norm_wave
            calc['scene'][i]['spectrum']['normalization']['norm_waveunit'] = 'microns'
            calc['scene'][i]['spectrum']['normalization']['type'] = 'at_lambda'

            # calculate position. NB this returns the center of the pixel
            ra, dec = lens.pixel_grid.map_pix2coord(x=item_number, y=row_number)

            # set position
            calc['scene'][i]['position']['x_offset'] = dec
            calc['scene'][i]['position']['y_offset'] = -ra
            # TODO clean up after confirming the above works
            # calc['scene'][i]['position']['x_offset'] = (item_number * (1 / 9) * (
            #         1 / oversample_factor)) + lens.ra_at_xy_0  # arcsec
            # calc['scene'][i]['position']['y_offset'] = (row_number * (1 / 9) * (
            #         1 / oversample_factor)) + lens.dec_at_xy_0  # arcsec

            i += 1

    if not suppress_output:
        print(f'Point source conversion complete: placed {i} point sources')

    return calc, i
