import os
import pytest

from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage


def test_roman_imaging():
    """
    Simplest case
    """
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

    assert synthetic_image.image is not None
    assert synthetic_image.lens_surface_brightness is None
    assert synthetic_image.source_surface_brightness is None

    assert synthetic_image.pixel_scale == 0.022
    assert synthetic_image.native_pixel_scale == 0.11
    assert synthetic_image.num_pix == 235
    assert synthetic_image.native_num_pix == 47
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.image.shape == (235, 235)

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    assert exposure.exposure is not None
    assert exposure.lens_exposure is None
    assert exposure.source_exposure is None

    # TODO checks on the images


def test_roman_generate_psf():
    """
    Provide a detector position by overriding the default instrument params where the PSF at that position is not cached, so the WebbPSF engine is automatically called to generate it
    """
    roman = Roman()
    lens = SampleStrongLens()
    band = 'F129'
    scene_size = 5  # arcsec
    oversample = 5
    exposure_time = 146
    instrument_params = {
        'detector': 1,
        'detector_position': (2047, 2047)
    }

    synthetic_image = SyntheticImage(strong_lens=lens,
                                     instrument=roman,
                                     band=band,
                                     arcsec=scene_size,
                                     oversample=oversample,
                                     instrument_params=instrument_params,
                                     verbose=False)

    with pytest.warns(UserWarning, match='PSF .* not found in cache .*'):
        exposure = Exposure(synthetic_image,
                            exposure_time=exposure_time,
                            verbose=False)

    # TODO checks on the images


def test_roman_cached_psf():
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

    import numpy as np
    psf = np.load(os.path.abspath('tests/test_data/F129_1_2048_2048_5_101.npy'))

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        psf=psf,
                        verbose=False)

    # TODO checks on the images


def test_roman_sky_background_off():
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

    engine_params = {
        'sky_background': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_all_detector_effects_off():
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

    engine_params = {
        'detector_effects': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_poisson_noise_off():
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

    engine_params = {
        'poisson_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_reciprocity_failure_off():
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

    engine_params = {
        'reciprocity_failure': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_dark_noise_off():
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

    engine_params = {
        'dark_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_nonlinearity_off():
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

    engine_params = {
        'nonlinearity': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_ipc_off():
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

    engine_params = {
        'ipc': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_read_noise_off():
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

    engine_params = {
        'read_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    # TODO checks on the images


def test_roman_pieces():
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
                                     pieces=True,
                                     verbose=False)

    assert synthetic_image.image is not None
    assert synthetic_image.lens_surface_brightness is not None
    assert synthetic_image.source_surface_brightness is not None

    exposure = Exposure(synthetic_image,
                        exposure_time=exposure_time,
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    assert exposure.exposure is not None
    assert exposure.lens_exposure is not None
    assert exposure.source_exposure is not None

    # TODO checks on the images
