import os
import galsim
import numpy as np

import mejiro
from mejiro.engines import galsim_engine
from mejiro.exposure import Exposure
from mejiro.instruments.hwo import HWO
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage


TEST_DATA_DIR = os.path.join(os.path.dirname(mejiro.__path__[0]), 'tests', 'test_data')


def test_hwo_default_engine_params():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        # don't provide engine params
                        verbose=False)

    assert exposure.engine == 'galsim'

    for key, item in exposure.engine_params.items():
        assert item == galsim_engine.default_engine_params('HWO')[key]


def test_hwo_pieces():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     pieces=True,
                                     verbose=False)

    assert synthetic_image.image.shape == (291, 291)
    assert synthetic_image.lens_surface_brightness.shape == (291, 291)
    assert synthetic_image.source_surface_brightness.shape == (291, 291)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        verbose=False)

    assert exposure.exposure.shape == (291, 291)
    assert exposure.lens_exposure.shape == (291, 291)
    assert exposure.source_exposure.shape == (291, 291)

    # TODO checks on the images


def test_hwo_noise():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        verbose=False)

    poisson_noise = exposure.poisson_noise
    dark_noise = exposure.dark_noise
    read_noise = exposure.read_noise

    assert type(poisson_noise) is galsim.Image
    assert type(dark_noise) is galsim.Image
    assert type(read_noise) is galsim.Image

    engine_params = {
        'poisson_noise': poisson_noise,
        'dark_noise': dark_noise,
        'read_noise': read_noise
    }

    exposure2 = Exposure(synthetic_image,
                         exposure_time=1000,
                         engine='galsim',
                         engine_params=engine_params,
                         verbose=False)

    assert np.array_equal(exposure2.exposure, exposure.exposure)
    assert np.array_equal(poisson_noise.array, exposure2.poisson_noise.array)
    assert np.array_equal(dark_noise.array, exposure2.dark_noise.array)
    assert np.array_equal(read_noise.array, exposure2.read_noise.array)


def test_hwo_sky_background_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    engine_params = {
        'sky_background': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_hwo_all_detector_effects_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    engine_params = {
        'detector_effects': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_hwo_poisson_noise_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    engine_params = {
        'poisson_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_hwo_dark_noise_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    engine_params = {
        'dark_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images


def test_hwo_read_noise_off():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=HWO(),
                                     band='J',
                                     verbose=False)

    engine_params = {
        'read_noise': False
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=1000,
                        engine='galsim',
                        engine_params=engine_params,
                        verbose=False)

    # TODO checks on the images