import os
import pytest
import galsim
import numpy as np
import os
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.instruments.roman import Roman
from mejiro.engines.stpsf_engine import STPSFEngine


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(mejiro.__file__)), 'tests', 'test_data')


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_exposure_with_galsim_engine(strong_lens):
    from mejiro.engines.galsim_engine import GalSimEngine

    band = 'F129'
    detector = 'SCA01'
    detector_position = (2048, 2048)

    kwargs_psf = STPSFEngine.get_roman_psf_kwargs(band, detector, detector_position, oversample=5, num_pix=101, check_cache=True, psf_cache_dir=TEST_DATA_DIR, verbose=False)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band=band,
                                     fov_arcsec=5,
                                     instrument_params={'detector': detector, 'detector_position': detector_position},
                                     kwargs_numerics={},
                                     kwargs_psf=kwargs_psf,
                                     pieces=False,
                                     verbose=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        verbose=False)

    assert exposure.synthetic_image == synthetic_image
    assert exposure.exposure_time == 146
    assert exposure.engine == 'galsim'
    assert exposure.verbose == False

    # check engine param defaulting
    ignored_keys = ['rng']
    for key, item in exposure.engine_params.items():
        if key not in ignored_keys:
            assert item == GalSimEngine.default_roman_engine_params()[key]

    # noise
    assert exposure.noise is not None
    assert type(exposure.noise) is np.ndarray
    assert exposure.noise.shape == (synthetic_image.num_pix, synthetic_image.num_pix)
    # engine-specific noise components are tested in the engine-specific tests

    # exposure
    assert type(exposure.exposure) is np.ndarray
    assert exposure.exposure.shape == (synthetic_image.num_pix, synthetic_image.num_pix)

    # image
    assert type(exposure.image) is galsim.Image
    assert exposure.image.array.shape == (synthetic_image.num_pix, synthetic_image.num_pix)

# TODO awaiting fix of lenstronomy engine
# def test_exposure_with_lenstronomy_engine():
#     from mejiro.engines import lenstronomy_engine

#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     exposure = Exposure(synthetic_image,
#                         exposure_time=146,
#                         engine='lenstronomy',
#                         # default engine params
#                         check_cache=True,
#                         psf_cache_dir=TEST_DATA_DIR,
#                         verbose=False)

#     assert exposure.synthetic_image == synthetic_image
#     assert exposure.exposure_time == 146
#     assert exposure.engine == 'lenstronomy'
#     assert exposure.verbose == False

#     # check engine param defaulting
#     assert exposure.engine_params == lenstronomy_engine.default_roman_engine_params()

#     # noise
#     assert exposure.noise is not None
#     assert type(exposure.noise) is np.ndarray
#     assert exposure.noise.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)
#     # engine-specific noise components are tested in the engine-specific tests

#     # exposure
#     assert type(exposure.exposure) is np.ndarray
#     assert exposure.exposure.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)

# TODO need to fix Pandeia engine for new version
# def test_exposure_with_pandeia_engine():
#     from mejiro.engines import pandeia_engine

#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     # for Pandeia engine, default of 10^4 takes almost 10 minutes to run, so reduce number of samples
#     engine_params = pandeia_engine.default_roman_engine_params()
#     engine_params['num_samples'] = 10

#     exposure = Exposure(synthetic_image,
#                         exposure_time=146,
#                         engine='pandeia',
#                         engine_params=engine_params,
#                         check_cache=True,
#                         psf_cache_dir=TEST_DATA_DIR,
#                         verbose=False)

#     assert exposure.synthetic_image == synthetic_image
#     assert exposure.exposure_time == 146
#     assert exposure.engine == 'pandeia'
#     assert exposure.verbose == False

#     # check engine param defaulting
#     assert exposure.engine_params == engine_params

#     # noise
#     assert exposure.noise is not None
#     assert type(exposure.noise) is np.ndarray
#     assert exposure.noise.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)
#     # engine-specific noise components are tested in the engine-specific tests

#     # exposure
#     assert type(exposure.exposure) is np.ndarray
#     assert exposure.exposure.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)

# TODO TEMP
# def test_default_engine():
#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     exposure = Exposure(synthetic_image,
#                         exposure_time=146,
#                         # don't provide engine
#                         check_cache=True,
#                         psf_cache_dir=TEST_DATA_DIR,
#                         verbose=False)

#     assert exposure.engine == 'galsim'


# def test_invalid_engine():
#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     try:
#         Exposure(synthetic_image,
#                  exposure_time=146,
#                  engine='invalid_engine',
#                  check_cache=True,
#                  psf_cache_dir=TEST_DATA_DIR,
#                  verbose=False)
#     except ValueError as e:
#         assert str(e) == 'Engine "invalid_engine" not recognized'


# def test_crop_edge_effects():
#     # unhappy path
#     image = np.zeros((100, 100))
#     with pytest.raises(AssertionError):
#         Exposure.crop_edge_effects(image)

#     # happy path
#     image = np.zeros((101, 101))
#     expected = np.zeros((98, 98))
#     result = Exposure.crop_edge_effects(image)
#     assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"
