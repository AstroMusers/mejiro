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
    detector = 1
    detector_position = (2048, 2048)
    exposure_time = 146

    synthetic_image = SyntheticImage(strong_lens=lens, 
                                     instrument=roman, 
                                     band=band, 
                                     arcsec=scene_size, 
                                     oversample=oversample, 
                                     sca=detector)
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time,
                        sca=detector,
                        sca_position=detector_position)
    
    # TODO these checks should go into a separate test for the SyntheticImage class
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
    detector = 1
    detector_position = (2048, 2048)
    exposure_time = 146

    synthetic_image = SyntheticImage(strong_lens=lens, 
                                     instrument=roman, 
                                     band=band, 
                                     arcsec=scene_size, 
                                     oversample=oversample, 
                                     sca=detector,
                                     sca_position=detector_position)
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time,
                        sca=detector,
                        sca_position=detector_position)
    
    # TODO these checks should go into a separate test for the SyntheticImage class
    assert synthetic_image.pixel_scale == 0.022
    assert synthetic_image.native_pixel_scale == 0.11
    assert synthetic_image.num_pix == 235
    assert synthetic_image.native_num_pix == 47
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.image.shape == (235, 235)
    