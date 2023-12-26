import random

from jwst_backgrounds import jbt
from regions import RectangleSkyRegion
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
import numpy as np

from mejiro.helpers import pandeia_input


def get_background():
    background = []

    wavelengths = get_wavelengths()
    ra, dec = generate_hlwas_coords()

    for wavelength in wavelengths:
        bg = jbt.background(ra, dec, wavelength)
        background.append(bg.bathtub['total_thiswave'][0])

    return [wavelengths, background]


def generate_hlwas_coords():
    return random.uniform(-60, -15), random.uniform(-60, -15)

    
def get_wavelengths():
    # roman_params = pandeia_input._get_roman_params()
    # min, _ = roman_params.get_min_max_wavelength('f106')
    # _, max = roman_params.get_min_max_wavelength('f184')
    # return np.arange(start=min, stop=max, step=0.1).tolist()
    return np.arange(start=0.5, stop=2.0, step=0.1).tolist()