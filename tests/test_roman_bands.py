import pytest

from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage


@pytest.mark.parametrize("band", ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213'])
def test_band(band):
    roman = Roman()
    lens = SampleStrongLens()
    scene_size = 5  # arcsec
    oversample = 5
    exposure_time = 146
    instrument_params = {
        'detector': 1,
        'detector_position': (2048, 2048)
    }

    synthetic_image = SyntheticImage(strong_lens=lens,
                                     instrument=roman,
                                     band=band,
                                     arcsec=scene_size,
                                     oversample=oversample,
                                     instrument_params=instrument_params,
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)

    assert synthetic_image.pixel_scale == 0.022
    assert synthetic_image.native_pixel_scale == 0.11
    assert synthetic_image.num_pix == 235
    assert synthetic_image.native_num_pix == 47
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.image.shape == (235, 235)

    # TODO checks on the images
