import pytest

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

@pytest.mark.parametrize("strong_lens", [SampleSL2S(), SampleBELLS()])
def test_lenstronomy_amplitudes(strong_lens):
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
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
