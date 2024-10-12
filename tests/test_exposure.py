from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage


def test_default_engine():
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
                        # don't provide engine
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)
    
    assert exposure.engine == 'galsim'
