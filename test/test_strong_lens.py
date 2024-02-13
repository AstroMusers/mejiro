import pytest

from mejiro.lenses.strong_lens import StrongLens
from mejiro.lenses.test import SampleStrongLens


def test_init():
    sample = SampleStrongLens()

    # check some attributes
    assert sample.z_lens == 0.643971
    assert sample.z_source == 1.627633

    # TODO check kwargs dicts

    # TODO verify model set up


# TODO test array
def test_get_array():
    sample = SampleStrongLens()

    num_pix = 45
    side = 4.95
    band = 'F184'
    kwargs_psf = {'psf_type': 'NONE'}

    array = sample.get_array(num_pix, side, band, kwargs_psf)

    assert array.shape == (num_pix, num_pix)
    