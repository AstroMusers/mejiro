from mejiro.exposure import Exposure
from mejiro.instruments.roman import Roman
from mejiro.lenses.test import SampleStrongLens
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import util
from mejiro.engines import galsim_engine, lenstronomy_engine, pandeia_engine


def test_exposure_with_galsim_engine():
    synthetic_image = util.unpickle('test_data/synthetic_image_roman_F129_5_5.pkl')

    # default engine params
    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim',
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)

    assert exposure.synthetic_image == synthetic_image
    assert exposure.exposure_time == 146
    assert exposure.engine == 'galsim'
    assert exposure.verbose == False

    # TODO exposure

    # TODO noise

    # TODO image


def test_exposure_with_lenstronomy_engine():
    synthetic_image = util.unpickle('test_data/synthetic_image_roman_F129_5_5.pkl')

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='lenstronomy',
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)

    assert exposure.synthetic_image == synthetic_image
    assert exposure.exposure_time == 146
    assert exposure.engine == 'lenstronomy'
    assert exposure.verbose == False


def test_exposure_with_pandeia_engine():
    synthetic_image = util.unpickle('test_data/synthetic_image_roman_F129_5_5.pkl')

    engine_params = {
        'num_samples': 100
    }

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='pandeia',
                        engine_params=engine_params,
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)

    assert exposure.synthetic_image == synthetic_image
    assert exposure.exposure_time == 146
    assert exposure.engine == 'pandeia'
    assert exposure.verbose == False


def test_default_engine():
    synthetic_image = util.unpickle('test_data/synthetic_image_roman_F129_5_5.pkl')

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        # don't provide engine
                        check_cache=True,
                        psf_cache_dir='test_data',
                        verbose=False)

    assert exposure.engine == 'galsim'


def test_invalid_engine():
    synthetic_image = util.unpickle('test_data/synthetic_image_roman_F129_5_5.pkl')

    try:
        Exposure(synthetic_image,
                 exposure_time=146,
                 engine='invalid_engine',
                 check_cache=True,
                 psf_cache_dir='test_data',
                 verbose=False)
    except ValueError as e:
        assert str(e) == 'Engine "invalid_engine" not recognized'
