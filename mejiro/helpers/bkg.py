import lenstronomy.Util.data_util as data_util
import numpy as np
import os
import random
from jwst_backgrounds import jbt
from scipy.stats import truncnorm

from mejiro.helpers import lenstronomy_sim
from mejiro.helpers.roman_params import RomanParameters
from mejiro.utils import util


def get_high_galactic_lat_bkg(shape, bands, seed=None):
    # was only one band provided as a string? or a list of bands?
    single_band = False
    if not isinstance(bands, list):
        single_band = True
        bands = [bands]

    # get a random multiplier which will vary zodiacal light between 1 and 2 times minimum, centered on 1.5 which is typical at high galactic latitudes
    loc = 1.5
    scale = 0.2
    clip_a = 1.
    clip_b = 2.
    a = (clip_a - loc) / scale
    b = (clip_b - loc) / scale
    dist = truncnorm(a=a, b=b, loc=loc, scale=scale)
    multiplier = dist.rvs(size=1)[0]

    # make an array out of this variance; it'll have positive and negative elements 
    if seed is not None:
        g = np.random.RandomState(seed=seed)
    else:
        g = np.random
    nx, ny = shape
    variance = np.zeros(shape)
    random_array = g.randn(nx, ny)

    backgrounds = []
    for band in bands:
        # get minimum zodical light in given band in counts/sec/pixel
        csv_path = os.path.join(_get_data_dir(), 'roman_spacecraft_and_instrument_parameters.csv')
        params = RomanParameters(csv_path)
        min_count_rate = params.get_min_zodi_count_rate(band)

        # construct an array for the baseline (uniform) sky background
        baseline = np.ones(shape) * min_count_rate * multiplier

        # calculate variance in this baseline
        kwargs_band = lenstronomy_sim.get_roman_band_kwargs(band)
        sigma_bkg = data_util.bkg_noise(readout_noise=0,
                                        exposure_time=kwargs_band['exposure_time'],
                                        sky_brightness=kwargs_band['sky_brightness'],
                                        pixel_scale=kwargs_band['pixel_scale'], num_exposures=1)
        variance += (random_array * sigma_bkg)

        # add baseline and variance
        backgrounds.append(baseline + variance)

    if single_band:
        return backgrounds[0]
    else:
        return backgrounds


def get_jbt_bkg(suppress_output=False):
    background = []

    wavelengths = get_wavelengths()
    ra, dec = generate_hlwas_coords()
    if not suppress_output:
        print(f'RA: {ra}, DEC: {dec}')

    for wavelength in wavelengths:
        bg = jbt.background(ra, dec, wavelength)
        background.append(bg.bathtub['total_thiswave'][0])

    return [wavelengths, background]


def generate_hlwas_coords():
    return random.uniform(15, 45), random.uniform(-45, -15)


def get_wavelengths():
    # roman_params = pandeia_input._get_roman_params()
    # min, _ = roman_params.get_min_max_wavelength('f106')
    # _, max = roman_params.get_min_max_wavelength('f184')
    # return np.arange(start=min, stop=max, step=0.1).tolist()
    return np.arange(start=0.5, stop=2.2, step=0.1).tolist()


# TODO does this function work properly?
def get_pandeia_randomized_bkg(lens, band):
    # load pre-generated Pandeia minzodi background
    data_dir = _get_data_dir()
    bkg = np.load(os.path.join(data_dir, f'pandeia_bkg_minzodi_benchmark_{band}.npy'))

    # crop and randomize
    bkg_cropped = util.center_crop_image(bkg, (lens.num_pix, lens.num_pix))
    flat = bkg_cropped.flatten()
    np.random.shuffle(flat)
    shuffled = flat.reshape(bkg_cropped.shape)

    # TODO account for different dimension

    return shuffled


def _get_data_dir():
    import mejiro
    module_path = os.path.dirname(mejiro.__file__)
    return os.path.join(module_path, 'data')
