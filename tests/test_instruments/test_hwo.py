import numpy as np
import pytest

from mejiro.instruments.hwo import HWO


def test_init():
    hwo = HWO()

    assert hwo.name == 'HWO'
    assert type(hwo.bands) == list
    assert type(hwo.engines) == list and len(hwo.engines) > 0

    # check values imported from hwo-tools
    assert type(hwo.aperture) == np.float64
    assert type(hwo.pivotwave) == dict
    # assert type(hwo.ab_zeropoint) == dict
    assert type(hwo.aperture_correction) == dict
    assert type(hwo.bandpass_r) == dict
    assert type(hwo.derived_bandpass) == dict


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
