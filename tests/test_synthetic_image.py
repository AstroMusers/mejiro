import pytest
from copy import deepcopy

from mejiro.instruments.roman import Roman
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import util


def test_magnitudes():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)

def test_lenstronomy_amplitudes():
    synthetic_image = SyntheticImage(strong_lens=SampleSL2S(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)


    synthetic_image = SyntheticImage(strong_lens=SampleBELLS(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)


def test_kwargs_numerics():
    # test defaulting
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     # none provided
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)
    
    # regular compute mode
    kwargs_numerics = {
        'supersampling_factor': 3,
        'compute_mode': 'regular'
    }
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)

    # adaptive compute mode with supersampled indices provided
    region = util.create_centered_circle(N=47, radius=10)
    kwargs_numerics = {
        'supersampling_factor': 3,
        'compute_mode': 'adaptive',
        'supersampled_indexes': region
    }
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)

    # adaptive compute mode with default supersampled indices (annulus around image positions)
    kwargs_numerics = {
        'supersampling_factor': 3,
        'compute_mode': 'adaptive',
    }
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01'},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)

def test_set_up_pixel_grid():
    strong_lens = SampleStrongLens()
    instrument = Roman()

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=instrument,
                                     band='F129',
                                     arcsec=4.95,
                                     oversample=5,
                                     verbose=False)
    assert synthetic_image.arcsec == 4.95
    assert synthetic_image.num_pix == 225

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=instrument,
                                     band='F129',
                                     arcsec=5.0,
                                     oversample=5,
                                     verbose=False)
    assert synthetic_image.arcsec == 5.17
    assert synthetic_image.num_pix == 235

    # warning paths
    with pytest.warns(UserWarning,
                      match='Oversampling factor less than 5 may not be sufficient for accurate results, especially when convolving with a non-trivial PSF'):
        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                         instrument=instrument,
                                         band='F129',
                                         arcsec=4.95,
                                         oversample=1,
                                         verbose=False)  # 4.95 / 0.11 = 45 so expect 45 pixels, then 4.95 arcsec
        assert synthetic_image.arcsec == 4.95
        assert synthetic_image.num_pix == 45

        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                         instrument=instrument,
                                         band='F129',
                                         arcsec=5.0,
                                         oversample=1,
                                         verbose=False)  # 5 / 0.11 = 45.45 so expect 46 pixels, then 47 so odd, then 5.17 arcsec
        assert synthetic_image.arcsec == 5.17
        assert synthetic_image.num_pix == 47

    # unhappy paths
    with pytest.raises(AssertionError):
        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                         instrument=instrument,
                                         band='F129',
                                         arcsec=5.0,
                                         oversample=2,
                                         verbose=False)  # even oversampling should raise an error

    with pytest.raises(AssertionError):
        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                         instrument=instrument,
                                         band='F129',
                                         arcsec=5.0,
                                         oversample=0.1,
                                         verbose=False)