import pytest

from mejiro.instruments.hwo import HWO
from mejiro.synthetic_image import SyntheticImage
from mejiro.exposure import Exposure
from mejiro.lenses.test import SampleStrongLens

def test_hwo_imaging():
    hwo = HWO()

    lens = SampleStrongLens()
    band = 'J'
    scene_size = 5  # arcsec
    oversample = 1
    exposure_time = 1000

    synthetic_image = SyntheticImage(strong_lens=lens, 
                                     instrument=hwo, 
                                     band=band, 
                                     arcsec=scene_size, 
                                     oversample=oversample,
                                     verbose=False)
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time)
    
    assert synthetic_image.pixel_scale == 0.04
    assert synthetic_image.native_pixel_scale == 0.04
    assert synthetic_image.num_pix == 125
    assert synthetic_image.native_num_pix == 125
    assert synthetic_image.arcsec == 5.0
    assert synthetic_image.image.shape == (125, 125)
    

def test_hwo_oversampled_imaging():
    hwo = HWO()

    lens = SampleStrongLens()
    band = 'J'
    scene_size = 5  # arcsec
    oversample = 5
    exposure_time = 1000

    synthetic_image = SyntheticImage(strong_lens=lens, 
                                     instrument=hwo, 
                                     band=band, 
                                     arcsec=scene_size, 
                                     oversample=oversample,
                                     verbose=False)
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time)
    
    assert synthetic_image.pixel_scale == 0.008
    assert synthetic_image.native_pixel_scale == 0.04
    assert synthetic_image.num_pix == 625
    assert synthetic_image.native_num_pix == 125
    assert synthetic_image.arcsec == 5.0
    assert synthetic_image.image.shape == (625, 625)
    