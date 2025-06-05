import os
import galsim
import numpy as np

import mejiro
from mejiro.engines import galsim_engine
from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage


TEST_DATA_DIR = os.path.join(os.path.dirname(mejiro.__path__[0]), 'tests', 'test_data')


def test_roman_default_engine_params():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        # don't provide engine params
                        verbose=False)

    assert exposure.engine == 'galsim'

    for key, item in exposure.engine_params.items():
        assert item == galsim_engine.default_engine_params('Roman')[key]


def test_roman_pieces():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     pieces=True,
                                     verbose=False)

    assert synthetic_image.image.shape == (47, 47)
    assert synthetic_image.lens_surface_brightness.shape == (47, 47)
    assert synthetic_image.source_surface_brightness.shape == (47, 47)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        verbose=False)

    assert exposure.exposure.shape == (47, 47)
    assert exposure.lens_exposure.shape == (47, 47)
    assert exposure.source_exposure.shape == (47, 47)

    # TODO checks on the images


def test_roman_noise():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        verbose=False)

    poisson_noise = exposure.poisson_noise
    reciprocity_failure = exposure.reciprocity_failure
    dark_noise = exposure.dark_noise
    nonlinearity = exposure.nonlinearity
    ipc = exposure.ipc
    read_noise = exposure.read_noise

    assert type(poisson_noise) is galsim.Image
    assert type(reciprocity_failure) is galsim.Image
    assert type(dark_noise) is galsim.Image
    assert type(nonlinearity) is galsim.Image
    assert type(ipc) is galsim.Image
    assert type(read_noise) is galsim.Image

    engine_params = {
        'poisson_noise': poisson_noise,
        'reciprocity_failure': reciprocity_failure,
        'dark_noise': dark_noise,
        'nonlinearity': nonlinearity,
        'ipc': ipc,
        'read_noise': read_noise
    }

    exposure2 = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    assert np.array_equal(exposure2.exposure, exposure.exposure)
    assert np.array_equal(poisson_noise.array, exposure2.poisson_noise.array)
    assert np.array_equal(reciprocity_failure.array, exposure2.reciprocity_failure.array)
    assert np.array_equal(dark_noise.array, exposure2.dark_noise.array)
    assert np.array_equal(nonlinearity.array, exposure2.nonlinearity.array)
    assert np.array_equal(ipc.array, exposure2.ipc.array)
    assert np.array_equal(read_noise.array, exposure2.read_noise.array)


def test_roman_sky_background_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'sky_background': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_all_detector_effects_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'detector_effects': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_poisson_noise_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'poisson_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_reciprocity_failure_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'reciprocity_failure': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_dark_noise_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'dark_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_nonlinearity_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'nonlinearity': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_ipc_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'ipc': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_roman_read_noise_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     verbose=False)

    engine_params = {
        'read_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images
