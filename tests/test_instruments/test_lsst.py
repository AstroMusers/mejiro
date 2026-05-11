import pytest
from astropy.units import Quantity

from mejiro.instruments.lsst import LSST


def test_init():
    lsst = LSST()

    # inherited attributes
    assert lsst.name == 'LSST'
    assert lsst.bands == ['u', 'g', 'r', 'i', 'z', 'y']
    assert isinstance(lsst.engines, list) and 'lenstronomy' in lsst.engines
    assert 'lenstronomy' in lsst.versions

    # direct attributes
    assert lsst.stray_light_fraction == 0.0
    assert lsst.gain is not None      # sourced from lenstronomy LSST camera dict
    assert lsst.pixel_scale is not None
    assert isinstance(lsst.lenstronomy_band_obs, dict)
    assert set(lsst.lenstronomy_band_obs.keys()) == set(lsst.bands)


def test_get_pixel_scale():
    lsst = LSST()
    assert lsst.get_pixel_scale('r') == Quantity(0.2, 'arcsec / pix')
    # band-independent
    assert lsst.get_pixel_scale('u') == lsst.get_pixel_scale('y')


def test_get_psf_fwhm_per_band():
    lsst = LSST()
    # seeing values come from lenstronomy's LSST config; assert structure + units
    for band in lsst.bands:
        fwhm = lsst.get_psf_fwhm(band)
        assert fwhm is not None
        assert fwhm.unit.is_equivalent('arcsec')
        assert fwhm.value > 0


def test_get_thermal_background_is_zero():
    lsst = LSST()
    for band in lsst.bands:
        assert lsst.get_thermal_background(band) == Quantity(0.0, 'ct / pix / s')


def test_get_dark_current_and_read_noise_are_band_independent():
    lsst = LSST()
    for band in lsst.bands:
        assert lsst.get_dark_current(band) == Quantity(0.0, 'ct / pix / s')
        assert lsst.get_read_noise(band) == Quantity(10.0, 'ct / pix / s')


def test_get_gain_returns_camera_value():
    lsst = LSST()
    # get_gain is band-independent and returns the lenstronomy camera ccd_gain
    assert lsst.get_gain('r') == lsst.gain
    assert lsst.get_gain('u') == lsst.get_gain('y')


def test_get_zeropoint_magnitude_per_band():
    lsst = LSST()
    # zeropoints come from lenstronomy LSST observation config; require positive AB mags
    for band in lsst.bands:
        zp = lsst.get_zeropoint_magnitude(band)
        assert zp is not None
        assert zp > 0


def test_default_params_and_validate():
    lsst = LSST()
    assert LSST.default_params() == {}
    sample = {'foo': 1, 'bar': 'baz'}
    assert lsst.validate_instrument_params(sample) == sample


def test_load_speclite_filters():
    speclite_filters = LSST.load_speclite_filters()
    assert speclite_filters is not None
    assert len(list(speclite_filters)) > 0
