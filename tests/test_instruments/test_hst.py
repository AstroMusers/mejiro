import os
import re

import pytest
from astropy.units import Quantity

import mejiro
from mejiro.instruments.hst import HST, ZEROPOINTS

SKYPY_CONFIG = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'skypy', 'hst', 'hst.yml')


def test_init():
    hst = HST()
    assert hst.name == 'HST'
    assert hst.bands == ['F438W', 'F475W', 'F606W', 'F814W']
    assert hst.num_detectors == 1
    assert hst.engines == ['galsim']


def test_every_band_has_its_parameters():
    hst = HST()
    for band in hst.bands:
        assert 24 < hst.get_zeropoint_magnitude(band) < 27
        assert hst.get_psf_fwhm(band).unit == 'arcsec'
        assert hst.get_pixel_scale(band) == Quantity(0.04, 'arcsec / pix')


def test_backgrounds_are_in_the_units_galsim_needs():
    # GalSimEngine.get_sky_background raises unless these are counts per pixel (per second)
    hst = HST()
    for band in hst.bands:
        assert hst.get_sky_level(band).unit == 'ct / pix'
        assert hst.get_thermal_background(band).unit == 'ct / pix'
    assert hst.get_dark_current('F606W').unit == 'ct / (pix s)'


def test_psf_is_a_gaussian_with_the_band_fwhm():
    hst = HST()
    # the pipeline passes Roman-style PSF arguments to every instrument; HST ignores them
    kwargs_psf = hst.get_psf_kwargs('F814W', oversample=1, num_pix=101, check_cache=True)
    assert kwargs_psf == {'psf_type': 'GAUSSIAN', 'fwhm': 0.074}


def test_speclite_filters_match_the_bands():
    names = HST.load_speclite_filters().names
    assert names == [f'WFC3_UVIS-{band}' for band in HST().bands]


def test_skypy_config_has_every_band():
    # step 01a draws magnitudes for the filters in the SkyPy config, and slsim looks them up as mag_<band>
    text = open(SKYPY_CONFIG).read()
    filters = re.search(r'^filters: \[(.*)\]$', text, re.M).group(1)
    assert [name.strip(" '") for name in filters.split(',')] == [f'WFC3_UVIS-{band}' for band in HST().bands]
    for columns in re.findall(r'^\s+(mag_\w+(?:, mag_\w+)*): !skypy', text, re.M):
        assert columns.split(', ') == [f'mag_{band}' for band in HST().bands]


def test_zeropoints_match_stsynphot():
    # the stored zeropoints were computed with stsynphot; check them wherever its data is available
    stsyn = pytest.importorskip('stsynphot')
    import numpy as np
    if not os.environ.get('PYSYN_CDBS'):
        pytest.skip('stsynphot needs STScI throughput tables (PYSYN_CDBS)')
    for band, zeropoint in ZEROPOINTS.items():
        bp = stsyn.band(f'wfc3,uvis1,{band.lower()},mjd#60035,aper#6.0')
        stmag = -21.1 - 2.5 * np.log10(bp.unit_response(stsyn.conf.area).value)
        assert stmag - 5 * np.log10(bp.pivot().value) + 18.6921 == pytest.approx(zeropoint, abs=1e-4)
