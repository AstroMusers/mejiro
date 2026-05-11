import pytest
from astropy.units import Quantity

from mejiro.instruments.hst import HST


def test_init():
    hst = HST()

    # inherited attributes
    assert hst.name == 'HST'
    assert hst.bands == ['F438W', 'F475W', 'F606W', 'F814W']
    assert isinstance(hst.engines, list) and len(hst.engines) > 0
    assert 'lenstronomy' in hst.versions

    # direct attributes
    assert hst.gain == 2.5
    assert hst.stray_light_fraction == 0.1
    assert isinstance(hst.dark_current, dict) and len(hst.dark_current) > 0
    assert isinstance(hst.read_noise, dict) and len(hst.read_noise) > 0
    assert isinstance(hst.psf_fwhm, dict) and len(hst.psf_fwhm) > 0
    assert isinstance(hst.thermal_background, dict) and len(hst.thermal_background) > 0
    assert isinstance(hst.sky_level, dict) and len(hst.sky_level) > 0


def test_get_pixel_scale():
    hst = HST()
    assert hst.get_pixel_scale('F438W') == Quantity(0.04, 'arcsec / pix')
    # documented as band-independent (returns the same scalar regardless of band)
    assert hst.get_pixel_scale('F814W') == hst.get_pixel_scale('F438W')


def test_get_psf_fwhm():
    hst = HST()
    # values come straight from the docstring-cited HST UVIS optical-performance table
    assert hst.get_psf_fwhm('F438W') == Quantity(0.070, 'arcsec')
    assert hst.get_psf_fwhm('F814W') == Quantity(0.074, 'arcsec')
    assert hst.get_psf_fwhm('F606W') == Quantity(0.067, 'arcsec')


def test_get_thermal_background_is_zero_for_uvis_bands():
    hst = HST()
    # "thermal background is negligible below ~8000 Angstrom" per the source comment
    for band in hst.bands:
        assert hst.get_thermal_background(band) == Quantity(0.0, 'ct / pix')


def test_get_sky_level():
    hst = HST()
    assert hst.get_sky_level('F438W') == Quantity(0.25, 'ct / pix')
    assert hst.get_sky_level('F814W') == Quantity(0.25, 'ct / pix')


def test_get_dark_current_and_read_noise():
    hst = HST()
    assert hst.get_dark_current('F438W') == Quantity(0.00319, 'ct / pix / s')
    assert hst.get_read_noise('F438W') == Quantity(3.0, 'ct / pix / s')


def test_get_gain_is_band_independent():
    hst = HST()
    assert hst.get_gain('F438W') == 2.5
    assert hst.get_gain('F814W') == 2.5


def test_build_obsmode_format():
    obsmode = HST.build_obsmode('uvis1', 'f438w', '60035', '6.0')
    assert obsmode == 'wfc3,uvis1,f438w,mjd#60035,aper#6.0'


def test_default_params_and_validate():
    hst = HST()
    assert HST.default_params() == {}
    # validate_instrument_params is a passthrough
    sample = {'foo': 1, 'bar': 'baz'}
    assert hst.validate_instrument_params(sample) == sample


def test_load_speclite_filters():
    speclite_filters = HST.load_speclite_filters()
    assert speclite_filters is not None
    # the loader globs WFC3_UVIS-*.ecsv under mejiro/data/hst_filter_response;
    # if the directory exists at all, at least one filter should be returned
    assert len(list(speclite_filters)) > 0
