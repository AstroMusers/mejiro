import pytest

from mejiro.exposure import Exposure
from mejiro.instruments.hwo import HWO
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage


@pytest.mark.parametrize("band", ['B', 'FUV', 'H', 'I', 'J', 'K', 'NUV', 'R', 'U', 'V'])
def test_band(band):
    hwo = HWO()
    lens = SampleStrongLens()
    scene_size = 5  # arcsec
    oversample = 5
    exposure_time = 100

    synthetic_image = SyntheticImage(strong_lens=lens,
                                     instrument=hwo,
                                     band=band,
                                     arcsec=scene_size,
                                     oversample=oversample,
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        verbose=False)

    # TODO account for different pixel scales for different filters
    # assert synthetic_image.pixel_scale == 0.008
    # assert synthetic_image.native_pixel_scale == 0.04
    # assert synthetic_image.num_pix == 625
    # assert synthetic_image.native_num_pix == 125
    # assert synthetic_image.arcsec == 5.0
    # assert synthetic_image.image.shape == (625, 625)

    # TODO checks on the images
