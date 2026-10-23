import pytest
from astropy.units import Quantity, UnitConversionError

from mejiro.instruments.instrument import Instrument
from mejiro.instruments.lsst import LSST


def test_init():
    lsst = LSST()

    # inherited attributes
    assert lsst.name == 'LSST'
    assert lsst.bands == ['u', 'g', 'r', 'i', 'z', 'y']
    assert isinstance(lsst.engines, list) and 'galsim' in lsst.engines
    assert 'chalcedon' in lsst.versions

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


def test_get_psf_kwargs_reuses_shared_gaussian_adapter():
    lsst = LSST()
    assert LSST.get_psf_kwargs is Instrument.get_psf_kwargs
    for band in lsst.bands:
        assert lsst.get_psf_kwargs(band, unused_option=True) == {
            'psf_type': 'GAUSSIAN', 'fwhm': lsst.get_psf_fwhm(band).value,
        }


def test_gaussian_psf_adapter_uses_band_getter_and_propagates_errors(monkeypatch):
    from unittest.mock import Mock

    lsst = LSST()
    getter = Mock(return_value=Quantity(0.12, 'arcsec'))
    monkeypatch.setattr(lsst, 'get_psf_fwhm', getter)
    assert lsst.get_psf_kwargs('fixture') == {'psf_type': 'GAUSSIAN', 'fwhm': 0.12}
    getter.assert_called_once_with('fixture')
    getter.side_effect = KeyError('missing band')
    with pytest.raises(KeyError, match='missing band'):
        lsst.get_psf_kwargs('missing')


def test_get_thermal_background_is_zero():
    lsst = LSST()
    for band in lsst.bands:
        assert lsst.get_thermal_background(band) == Quantity(0.0, 'ct / (pix s)')


def test_get_dark_current_and_read_noise_are_band_independent():
    lsst = LSST()
    for band in lsst.bands:
        assert lsst.get_dark_current(band) == Quantity(0.0, 'ct / (pix s)')
        assert lsst.get_read_noise(band) == Quantity(10.0, 'ct / (pix s)')


def test_get_noise_reuses_instrument_method():
    lsst = LSST()
    assert LSST.get_noise is Instrument.get_noise
    for band in lsst.bands:
        assert lsst.get_noise(band) == Quantity(10.0, 'ct / (pix s)')


def test_get_noise_preserves_band_dispatch_units_and_errors(monkeypatch):
    lsst = LSST()
    dark_current = {'r': Quantity(3.0, 'ct / (pix s)'), 'u': Quantity(5.0, 'ct / (pix s)')}
    read_noise = {'r': Quantity(0.004, 'ct / (pix ms)'), 'u': Quantity(0.006, 'ct / (pix ms)')}
    calls = []

    def get_dark_current(band):
        calls.append(('dark', band))
        return dark_current[band]

    def get_read_noise(band):
        calls.append(('read', band))
        return read_noise[band]

    monkeypatch.setattr(lsst, 'get_dark_current', get_dark_current)
    monkeypatch.setattr(lsst, 'get_read_noise', get_read_noise)
    assert lsst.get_noise('r') == Quantity(7.0, 'ct / (pix s)')
    assert lsst.get_noise('u') == Quantity(11.0, 'ct / (pix s)')
    assert calls == [('dark', 'r'), ('read', 'r'), ('dark', 'u'), ('read', 'u')]
    with pytest.raises(KeyError, match='missing'):
        lsst.get_noise('missing')
    assert calls[-1] == ('dark', 'missing')
    read_noise['r'] = Quantity(1.0, 'arcsec')
    with pytest.raises(UnitConversionError):
        lsst.get_noise('r')


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
