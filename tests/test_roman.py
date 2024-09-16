import pytest

from mejiro.instruments.roman import Roman


def test_init():
    roman = Roman()
    assert roman.name == 'Roman'
    assert roman.pixel_scale == 0.11
    assert roman.pixels_per_axis == 4088


def test_get_sca_string():
    roman = Roman()

    # test ints
    assert roman.get_sca_string(1) == 'SCA01'
    assert roman.get_sca_string(10) == 'SCA10'

    # test floats
    assert roman.get_sca_string(1.0) == 'SCA01'
    assert roman.get_sca_string(10.0) == 'SCA10'

    # test strings
    assert roman.get_sca_string('1') == 'SCA01'
    assert roman.get_sca_string('10') == 'SCA10'
    assert roman.get_sca_string('01') == 'SCA01'
    assert roman.get_sca_string('SCA01') == 'SCA01'
    assert roman.get_sca_string('SCA10') == 'SCA10'
    assert roman.get_sca_string('SCA1') == 'SCA01'


def test_get_sca_int():
    roman = Roman()

    # test ints
    assert roman.get_sca_int(1) == 1
    assert roman.get_sca_int(10) == 10

    # test floats
    assert roman.get_sca_int(1.0) == 1
    assert roman.get_sca_int(10.0) == 10

    # test strings
    assert roman.get_sca_int('1') == 1
    assert roman.get_sca_int('10') == 10
    assert roman.get_sca_int('01') == 1
    assert roman.get_sca_int('SCA01') == 1
    assert roman.get_sca_int('SCA10') == 10
    assert roman.get_sca_int('SCA1') == 1


def test_translate_band():
    roman = Roman()

    # test valid bands
    assert roman.translate_band('F062') == 'F062'
    assert roman.translate_band('F087') == 'F087'
    assert roman.translate_band('F106') == 'F106'
    assert roman.translate_band('F129') == 'F129'
    assert roman.translate_band('F158') == 'F158'
    assert roman.translate_band('F184') == 'F184'
    assert roman.translate_band('F146') == 'F146'
    assert roman.translate_band('F213') == 'F213'

    # test valid aliases
    assert roman.translate_band('R') == 'F062'
    assert roman.translate_band('Z') == 'F087'
    assert roman.translate_band('Y') == 'F106'
    assert roman.translate_band('J') == 'F129'
    assert roman.translate_band('H') == 'F158'
    assert roman.translate_band('H/K') == 'F184'
    assert roman.translate_band('WIDE') == 'F146'
    assert roman.translate_band('KS') == 'F213'

    # test invalid band
    try:
        roman.translate_band('X')
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # test invalid alias
    try:
        roman.translate_band('A')
        assert False, "Expected ValueError"
    except ValueError:
        pass
