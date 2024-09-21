import pytest

from mejiro.utils import roman_util


def test_divide_up_sca():
    # sides = 1
    result = roman_util.divide_up_sca(1)
    expected = [(2044, 2044)]
    assert result == expected, f"Expected {expected}, but got {result}"

    # sides = 2 (as float)
    result = roman_util.divide_up_sca(2.)
    expected = [(1022, 1022), (1022, 3066), (3066, 1022), (3066, 3066)]
    assert result == expected, f"Expected {expected}, but got {result}"

    # sides = 3
    result = roman_util.divide_up_sca(3)
    expected = [
        (681, 681),
        (681, 2044),
        (681, 3407),
        (2044, 681),
        (2044, 2044),
        (2044, 3407),
        (3407, 681),
        (3407, 2044),
        (3407, 3407)
    ]

    assert result == expected, f"Expected {expected}, but got {result}"

    # sides = 4
    result = roman_util.divide_up_sca(4)
    expected = [
        (511, 511), (511, 1533), (511, 2555), (511, 3577),
        (1533, 511), (1533, 1533), (1533, 2555), (1533, 3577),
        (2555, 511), (2555, 1533), (2555, 2555), (2555, 3577),
        (3577, 511), (3577, 1533), (3577, 2555), (3577, 3577)
    ]
    assert result == expected, f"Expected {expected}, but got {result}"

    # unhappy path
    with pytest.raises(AssertionError):
        roman_util.divide_up_sca(0)


def test_get_sca_string():
    # test ints
    assert roman_util.get_sca_string(1) == 'SCA01'
    assert roman_util.get_sca_string(10) == 'SCA10'

    # test floats
    assert roman_util.get_sca_string(1.0) == 'SCA01'
    assert roman_util.get_sca_string(10.0) == 'SCA10'

    # test strings
    assert roman_util.get_sca_string('1') == 'SCA01'
    assert roman_util.get_sca_string('10') == 'SCA10'
    assert roman_util.get_sca_string('01') == 'SCA01'
    assert roman_util.get_sca_string('SCA01') == 'SCA01'
    assert roman_util.get_sca_string('SCA10') == 'SCA10'
    assert roman_util.get_sca_string('SCA1') == 'SCA01'


def test_get_sca_int():
    # test ints
    assert roman_util.get_sca_int(1) == 1
    assert roman_util.get_sca_int(10) == 10

    # test floats
    assert roman_util.get_sca_int(1.0) == 1
    assert roman_util.get_sca_int(10.0) == 10

    # test strings
    assert roman_util.get_sca_int('1') == 1
    assert roman_util.get_sca_int('10') == 10
    assert roman_util.get_sca_int('01') == 1
    assert roman_util.get_sca_int('SCA01') == 1
    assert roman_util.get_sca_int('SCA10') == 10
    assert roman_util.get_sca_int('SCA1') == 1


def test_translate_band():
    # test valid bands
    assert roman_util.translate_band('F062') == 'F062'
    assert roman_util.translate_band('F087') == 'F087'
    assert roman_util.translate_band('F106') == 'F106'
    assert roman_util.translate_band('F129') == 'F129'
    assert roman_util.translate_band('F158') == 'F158'
    assert roman_util.translate_band('F184') == 'F184'
    assert roman_util.translate_band('F146') == 'F146'
    assert roman_util.translate_band('F213') == 'F213'

    # test valid aliases
    assert roman_util.translate_band('R') == 'F062'
    assert roman_util.translate_band('Z') == 'F087'
    assert roman_util.translate_band('Y') == 'F106'
    assert roman_util.translate_band('J') == 'F129'
    assert roman_util.translate_band('H') == 'F158'
    assert roman_util.translate_band('H/K') == 'F184'
    assert roman_util.translate_band('WIDE') == 'F146'
    assert roman_util.translate_band('KS') == 'F213'

    # test invalid band
    try:
        roman_util.translate_band('X')
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # test invalid alias
    try:
        roman_util.translate_band('A')
        assert False, "Expected ValueError"
    except ValueError:
        pass