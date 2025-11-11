import numpy as np
import pytest

from mejiro.instruments.hwo import HWO


def test_init():
    hwo = HWO()

    # super
    assert hwo.name == 'HWO'
    assert type(hwo.bands) == list
    assert type(hwo.engines) == list and len(hwo.engines) > 0

    # set attributes
    assert hwo.gain == {
        'B': 1.0,
        'FUV': 1.0,
        'H': 1.0,
        'I': 1.0,
        'J': 1.0,
        'K': 1.0,
        'NUV': 1.0,
        'R': 1.0,
        'U': 1.0,
        'V': 1.0,
    }
    assert hwo.stray_light_fraction == 0.01
    assert hwo.aperture is not None
    assert hwo.pixel_scale is not None
    assert hwo.dark_current is not None
    assert hwo.read_noise is not None
    assert hwo.sky_level is not None
    assert hwo.psf_fwhm is not None
    assert hwo.thermal_background is not None
    assert hwo.zeropoints is not None

def test_eacs():
    # test default
    hwo = HWO()
    assert hwo.telescope.name == 'HWO-EAC-1'

    # test setting
    hwo = HWO(eac='EAC1')
    assert hwo.telescope.name == 'HWO-EAC-1'

    hwo = HWO(eac='EAC2')
    assert hwo.telescope.name == 'HWO-EAC-2'

    hwos = HWO(eac='EAC3')
    assert hwos.telescope.name == 'HWO-EAC-3'

def test_get_pixel_scale():
    hwo = HWO()
    assert hwo.get_pixel_scale('J') is not None
    assert hwo.get_pixel_scale('J') != hwo.get_pixel_scale('I')

def test_get_zeropoint_magnitude():
    hwo = HWO()
    assert hwo.get_zeropoint_magnitude('J') is not None
    assert hwo.zeropoints is not None

def test_get_thermal_background():
    hwo = HWO()
    assert hwo.get_thermal_background('J') is not None
    assert hwo.thermal_background is not None

def test_get_psf_fwhm():
    hwo = HWO()
    assert hwo.get_psf_fwhm('J') is not None
    assert hwo.psf_fwhm is not None

def test_load_speclite_filters():
    speclite_filters = HWO.load_speclite_filters()
    assert speclite_filters is not None
