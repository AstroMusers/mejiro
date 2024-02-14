import pytest

from mejiro.helpers import gs
from mejiro.lenses.strong_lens import StrongLens
from mejiro.lenses.test import SampleStrongLens


def test_get_image():
    sample_lens = SampleStrongLens()

    num_pix = 45
    side = 4.95

    # TODO one band, one array (single band happy path)
    array = sample_lens.get_array(num_pix, side, 'F106')

    # TODO 3 bands, 3 arrays (color happy path)

    # TODO one band, multiple arrays

    # TODO two bands, any number of arrays

    # TODO multiple bands, one array

    # TODO any number of bands, two arrays
