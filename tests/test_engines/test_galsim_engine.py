from mejiro.engines import galsim_engine
from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage


def test_roman_defaulting_engine_params():
    roman = Roman()

    lens = SampleStrongLens()
    band = 'F129'
    scene_size = 5  # arcsec
    oversample = 5
    exposure_time = 146

    synthetic_image = SyntheticImage(strong_lens=lens,
                                     instrument=roman,
                                     band=band,
                                     arcsec=scene_size,
                                     oversample=oversample,
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)

    default_roman_engine_params = galsim_engine.default_roman_engine_params()
