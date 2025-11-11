import pytest
from astropy.units import Quantity

from mejiro.instruments.jwst import JWST


def test_init():
    jwst = JWST()

    # check inherited attributes (less strict than Roman-specific tests)
    assert jwst.name
    assert type(jwst.bands) == list and len(jwst.bands) > 0
    assert type(jwst.engines) == list and len(jwst.engines) > 0

    # check fields to initialize
    assert jwst.zeropoints is not None
    assert jwst.thermal_background is not None

def test_get_pixel_scale():
    jwst = JWST()
    assert jwst.get_pixel_scale('F115W') is not None

def test_get_zeropoint_magnitude():
    jwst = JWST()
    assert jwst.get_zeropoint_magnitude('F115W') is not None

def test_get_thermal_background():
    jwst = JWST()
    assert jwst.get_thermal_background('F115W') is not None

# def test_get_psf_fwhm():
#     jwst = JWST()
#     assert jwst.get_psf_fwhm('F115W') is not None
#     assert jwst.psf_fwhm is not None

def test_load_speclite_filters():
    speclite_filters = JWST.load_speclite_filters()
    assert speclite_filters is not None
