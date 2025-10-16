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
    assert hwo.stray_light_fraction == 0.1
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
