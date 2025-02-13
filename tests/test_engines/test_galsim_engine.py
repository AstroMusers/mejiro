import os
import galsim
import numpy as np
import os

import mejiro
from mejiro.engines import galsim_engine
from mejiro.exposure import Exposure
from mejiro.instruments.hwo import HWO
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage

TEST_DATA_DIR = os.path.join(os.path.dirname(mejiro.__path__[0]), 'tests', 'test_data')


def test_roman_default_engine_params():
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
                        engine='galsim',
                        # don't provide engine params
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    assert exposure.engine == 'galsim'

    for key, item in exposure.engine_params.items():
        assert item == galsim_engine.default_roman_engine_params()[key]


def test_roman_noise():
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
                        psf_cache_dir=os.path.abspath('tests/test_data'),
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
                         exposure_time=exposure_time,
                         engine_params=engine_params,
                         check_cache=True,
                         psf_cache_dir=os.path.abspath('tests/test_data'),
                         verbose=False)

    assert np.array_equal(exposure2.exposure, exposure.exposure)
    assert np.array_equal(poisson_noise.array, exposure2.poisson_noise.array)
    assert np.array_equal(reciprocity_failure.array, exposure2.reciprocity_failure.array)
    assert np.array_equal(dark_noise.array, exposure2.dark_noise.array)
    assert np.array_equal(nonlinearity.array, exposure2.nonlinearity.array)
    assert np.array_equal(ipc.array, exposure2.ipc.array)
    assert np.array_equal(read_noise.array, exposure2.read_noise.array)


def test_hwo_default_engine_params():
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
                        exposure_time=exposure_time,
                        engine='galsim',
                        # don't provide engine params
                        check_cache=True,
                        psf_cache_dir=os.path.abspath('tests/test_data'),
                        verbose=False)

    assert exposure.engine == 'galsim'

    for key, item in exposure.engine_params.items():
        assert item == galsim_engine.default_roman_engine_params()[key]


def test_hwo_noise():
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
                        exposure_time=exposure_time,
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
                         exposure_time=exposure_time,
                         engine_params=engine_params,
                         check_cache=True,
                         psf_cache_dir=os.path.abspath('tests/test_data'),
                         verbose=False)

    assert np.array_equal(exposure2.exposure, exposure.exposure)
    assert np.array_equal(poisson_noise.array, exposure2.poisson_noise.array)
    assert np.array_equal(dark_noise.array, exposure2.dark_noise.array)
    assert np.array_equal(read_noise.array, exposure2.read_noise.array)
