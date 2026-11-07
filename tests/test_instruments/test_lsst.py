import os
import re

from astropy.units import Quantity

import mejiro
from mejiro.instruments.lsst import LSST

SKYPY_CONFIG = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'skypy', 'lsst', 'lsst.yml')


def test_init():
    lsst = LSST()
    assert lsst.name == 'LSST'
    assert lsst.bands == ['u', 'g', 'r', 'i', 'z', 'y']
    assert lsst.num_detectors == 1
    assert lsst.engines == ['galsim']


def test_every_band_has_its_parameters():
    lsst = LSST()
    for band in lsst.bands:
        assert 25 < lsst.get_zeropoint_magnitude(band) < 30
        assert lsst.get_psf_fwhm(band).unit == 'arcsec' and lsst.get_psf_fwhm(band).value > 0
        assert lsst.get_pixel_scale(band) == Quantity(0.2, 'arcsec / pix')
        assert lsst.get_gain(band) > 0


def test_sky_is_brighter_in_the_red():
    # lenstronomy's sky brightness falls from 23 mag / arcsec^2 in u to 18.6 in y
    lsst = LSST()
    assert lsst.get_sky_level('y') > lsst.get_sky_level('u')


def test_backgrounds_are_in_the_units_galsim_needs():
    # GalSimEngine.get_sky_background raises unless these are counts per pixel (per second)
    lsst = LSST()
    for band in lsst.bands:
        assert lsst.get_sky_level(band).unit == 'ct / pix'
        assert lsst.get_thermal_background(band).unit == 'ct / pix'
    assert lsst.get_dark_current('r').unit == 'ct / (pix s)'


def test_psf_is_a_gaussian_with_the_band_seeing():
    lsst = LSST()
    # the pipeline passes Roman-style PSF arguments to every instrument; LSST ignores them
    kwargs_psf = lsst.get_psf_kwargs('i', oversample=1, num_pix=101, check_cache=True)
    assert kwargs_psf == {'psf_type': 'GAUSSIAN', 'fwhm': lsst.get_psf_fwhm('i').value}


def test_speclite_filters_match_the_bands():
    names = LSST.load_speclite_filters().names
    assert names == [f'lsst2023-{band}' for band in LSST().bands]


def test_skypy_config_has_every_band():
    # step 01a draws magnitudes for the filters in the SkyPy config, and slsim looks them up as mag_<band>
    text = open(SKYPY_CONFIG).read()
    filters = re.search(r'^filters: \[(.*)\]$', text, re.M).group(1)
    assert [name.strip(" '") for name in filters.split(',')] == [f'lsst2023-{band}' for band in LSST().bands]
    for columns in re.findall(r'^\s+(mag_\w+(?:, mag_\w+)*): !skypy', text, re.M):
        assert columns.split(', ') == [f'mag_{band}' for band in LSST().bands]
