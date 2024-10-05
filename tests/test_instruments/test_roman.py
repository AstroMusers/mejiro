import pytest

from mejiro.instruments.roman import Roman


def test_init():
    roman = Roman()

    assert roman.name == 'Roman'
    assert roman.pixels_per_axis == 4088

    # check that all files loaded
    assert not roman.df.empty, 'roman_spacecraft_and_instrument_parameters DataFrame is empty'
    assert roman.zp_dict, 'zp_dict is empty'  # NB empty dictionaries evaluate to False
    assert roman.min_zodi_dict, 'min_zodi_dict is empty'
    assert roman.thermal_bkg_dict, 'thermal_bkg_dict is empty'


def test_validate_instrument_params():
    roman = Roman()

    # Test valid parameters
    valid_params = {
        'detector': 5,
        'detector_position': (2044, 2044)
    }
    try:
        roman.validate_instrument_params(valid_params)
    except AssertionError:
        pytest.fail("validate_instrument_params raised AssertionError unexpectedly with valid parameters!")

    # Test invalid detector number
    invalid_detector_params = {
        'detector': 20,
        'detector_position': (2044, 2044)
    }
    with pytest.raises(AssertionError, match='Detector number must be an integer between 1 and 18.'):
        roman.validate_instrument_params(invalid_detector_params)

    # Test invalid detector position type
    invalid_position_type_params = {
        'detector': 5,
        'detector_position': [2044, 2044]
    }
    with pytest.raises(AssertionError, match='The detector_position parameter must be an \(x,y\) coordinate tuple.'):
        roman.validate_instrument_params(invalid_position_type_params)

    # Test invalid detector position length
    invalid_position_length_params = {
        'detector': 5,
        'detector_position': (2044,)
    }
    with pytest.raises(AssertionError, match='The detector_position parameter must be an \(x,y\) coordinate tuple.'):
        roman.validate_instrument_params(invalid_position_length_params)

    # Test invalid detector position values
    invalid_position_values_params = {
        'detector': 5,
        'detector_position': (5000, 2044)
    }
    with pytest.raises(AssertionError, match='Choose a valid pixel position on the range 4-4092.'):
        roman.validate_instrument_params(invalid_position_values_params)
