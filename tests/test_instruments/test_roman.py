import pytest
from astropy.units import Quantity

from mejiro.instruments.roman import Roman


def test_init():
    roman = Roman()

    # check inherited attributes
    assert roman.name == 'Roman'
    assert roman.bands == ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
    assert type(roman.engines) == list and len(roman.engines) > 0
    assert roman.versions

    # check other attributes
    assert roman.pixel_scale == Quantity(0.11, 'arcsec / pix')

    # check fields to initialize
    assert roman.zeropoints is None
    assert roman.thermal_background is None
    assert roman.minimum_zodiacal_light is None
    assert roman.psf_fwhm is None

def test_get_pixel_scale():
    roman = Roman()
    assert roman.get_pixel_scale() == Quantity(0.11, 'arcsec / pix')

def test_get_zeropoint_magnitude():
    roman = Roman()
    assert roman.get_zeropoint_magnitude('F062', 'SCA01') == pytest.approx(26.57, rel=1e-2)
    assert roman.zeropoints is not None
    
def test_get_minimum_zodiacal_light():
    roman = Roman()
    assert roman.get_minimum_zodiacal_light('F062') == Quantity(0.25, 'ct / pix')
    assert roman.minimum_zodiacal_light is not None

def test_get_thermal_background():
    roman = Roman()
    assert roman.get_thermal_background('F062') == Quantity(0.003, 'ct / pix')
    assert roman.thermal_background is not None

def test_get_psf_fwhm():
    roman = Roman()
    assert roman.get_psf_fwhm('F062') == Quantity(0.058, 'arcsec')
    assert roman.psf_fwhm is not None

def test_validate_instrument_params():
    roman = Roman()

    # Test valid parameters
    valid_params = {
        'detector': 5,
        'detector_position': (2044, 2044)
    }
    try:
        roman.validate_instrument_params(valid_params)
    except ValueError:
        pytest.fail("validate_instrument_params raised AssertionError unexpectedly with valid parameters!")

    # Test invalid detector number
    invalid_detector_params = {
        'detector': 20,
        'detector_position': (2044, 2044)
    }
    with pytest.raises(ValueError, match='Detector number must be an integer between 1 and 18.'):
        roman.validate_instrument_params(invalid_detector_params)

    # Test invalid detector position type
    invalid_position_type_params = {
        'detector': 5,
        'detector_position': [2044, 2044]
    }
    with pytest.raises(ValueError, match='The detector_position parameter must be an \(x,y\) coordinate tuple.'):
        roman.validate_instrument_params(invalid_position_type_params)

    # Test invalid detector position length
    invalid_position_length_params = {
        'detector': 5,
        'detector_position': (2044,)
    }
    with pytest.raises(ValueError, match='The detector_position parameter must be an \(x,y\) coordinate tuple.'):
        roman.validate_instrument_params(invalid_position_length_params)

    # Test invalid detector position values
    invalid_position_values_params = {
        'detector': 5,
        'detector_position': (5000, 2044)
    }
    with pytest.raises(ValueError, match='Choose a valid pixel position on the range 4-4092.'):
        roman.validate_instrument_params(invalid_position_values_params)
