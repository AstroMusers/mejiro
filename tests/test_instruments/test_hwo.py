import pytest
import numpy as np

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
    # assert type(hwo.derived_bandpass) == dict