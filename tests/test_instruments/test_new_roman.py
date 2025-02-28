import pytest

from mejiro.instruments.new_roman import Roman


RTI_PATH = '/nfsdata1/bwedig/STScI/roman-technical-information'


def test_init():
    roman = Roman(roman_technical_information_path=RTI_PATH)

    # check inherited attributes
    assert roman.name == 'Roman'
    # assert
    # roman.bands == ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
    assert type(roman.engines) == list and len(roman.engines) > 0

    # check other attributes
    assert roman.version is not None
    