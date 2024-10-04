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
    oversample = 5
    exposure_time = 1000

    synthetic_image = SyntheticImage(strong_lens=lens, 
                                    instrument=hwo, 
                                    band=band, 
                                    arcsec=scene_size, 
                                    oversample=oversample,
                                    verbose=False)
    
    assert synthetic_image.image is not None
    assert synthetic_image.lens_surface_brightness is None
    assert synthetic_image.source_surface_brightness is None

    assert synthetic_image.pixel_scale == 0.008
    assert synthetic_image.native_pixel_scale == 0.04
    assert synthetic_image.num_pix == 625
    assert synthetic_image.native_num_pix == 125
    assert synthetic_image.arcsec == 5.0
    assert synthetic_image.image.shape == (625, 625)
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time, 
                        verbose=False)
    
    assert exposure.exposure is not None
    assert exposure.lens_exposure is None
    assert exposure.source_exposure is None

    # TODO checks on the images
    

def test_hwo_sky_background_off():
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
    
    engine_params = {
        'sky_background': False
    }
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time, 
                        engine_params=engine_params,
                        verbose=False)
    
    # TODO checks on the images


def test_hwo_all_detector_effects_off():
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
    
    engine_params = {
        'detector_effects': False
    }
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time, 
                        engine_params=engine_params,
                        verbose=False)
    
    # TODO checks on the images


def test_hwo_poisson_noise_off():
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
    
    engine_params = {
        'poisson_noise': False
    }
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time, 
                        engine_params=engine_params,
                        verbose=False)
    
    # TODO checks on the images


def test_hwo_dark_noise_off():
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
    
    engine_params = {
        'dark_noise': False
    }
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time, 
                        engine_params=engine_params,
                        verbose=False)
    
    # TODO checks on the images


def test_hwo_read_noise_off():
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
    
    engine_params = {
        'read_noise': False
    }
    
    exposure = Exposure(synthetic_image, 
                        exposure_time=exposure_time, 
                        engine_params=engine_params,
                        verbose=False)
    
    # TODO checks on the images


def test_hwo_pieces():
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
                                     pieces=True,
                                     verbose=False)
    
    assert synthetic_image.image is not None
    assert synthetic_image.lens_surface_brightness is not None
    assert synthetic_image.source_surface_brightness is not None

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        verbose=False)
    
    assert exposure.exposure is not None
    assert exposure.lens_exposure is not None
    assert exposure.source_exposure is not None

    # TODO checks on the images
