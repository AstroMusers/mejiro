import pytest

from mejiro.instruments.roman import Roman
from mejiro.synthetic_image import SyntheticImage
from mejiro.exposure import Exposure
from mejiro.lenses.test import SampleStrongLens

def test_roman_imaging():
    roman = Roman()

    lens = SampleStrongLens()
    band = 'F129'
    scene_size = 5  # arcsec
    oversample = 1
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
                        exposure_time=exposure_time, verbose=False)
    
    assert synthetic_image.pixel_scale == 0.11
    assert synthetic_image.native_pixel_scale == 0.11
    assert synthetic_image.num_pix == 47
    assert synthetic_image.native_num_pix == 47
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.image.shape == (47, 47)
    

def test_roman_oversampled_imaging():
    roman = Roman()

    lens = SampleStrongLens()
    band = 'F129'
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
                        exposure_time=exposure_time, verbose=False)
    
    assert synthetic_image.pixel_scale == 0.022
    assert synthetic_image.native_pixel_scale == 0.11
    assert synthetic_image.num_pix == 235
    assert synthetic_image.native_num_pix == 47
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.image.shape == (235, 235)

# TODO test pieces
    