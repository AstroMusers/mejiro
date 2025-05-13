import pytest
import numpy as np

from mejiro.instruments.roman import Roman
from mejiro.galaxy_galaxy import SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage
from mejiro.utils import util


def test_magnitudes():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
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
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
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
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     # none provided
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)
    assert synthetic_image.kwargs_numerics == SyntheticImage.DEFAULT_KWARGS_NUMERICS
    
    # regular compute mode
    kwargs_numerics = {
        'supersampling_factor': 5,
        'compute_mode': 'regular'
    }
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)

    # adaptive compute mode with supersampled indices provided
    region = util.create_centered_circle(N=47, radius=10)
    kwargs_numerics = {
        'supersampling_factor': 5,
        'compute_mode': 'adaptive',
        'supersampled_indexes': region
    }
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)

    # adaptive compute mode with default supersampled indices (annulus around image positions)
    kwargs_numerics = {
        'supersampling_factor': 5,
        'compute_mode': 'adaptive',
    }
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)
    
    # unhappy path: insufficient supersampling factor
    kwargs_numerics = {
        'supersampling_factor': 1,
        'compute_mode': 'adaptive',
    }
    with pytest.warns(UserWarning,
                      match='Supersampling factor less than 5 may not be sufficient for accurate results, especially when convolving with a non-trivial PSF'):
        synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=True)
        

@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])        
def test_build_adaptive_grid(strong_lens):
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)
    
    image_positions = synthetic_image.get_image_positions(pixel=True)

    # check that with a padding of 1 pixel, the grid always includes all image positions
    includes_grid = synthetic_image.build_adaptive_grid(pad=1)
    for i, _ in enumerate(image_positions[0]):
        assert includes_grid[round(image_positions[1][i]), round(image_positions[0][i])]

    # check that a huge grid is slimmed down to the size of the image
    huge_grid = synthetic_image.build_adaptive_grid(pad=1000)
    assert huge_grid.shape[0] == synthetic_image.image.shape[0]
    assert huge_grid.shape[1] == synthetic_image.image.shape[1]

    # unhappy path: negative padding
    with pytest.raises(ValueError, match='Padding value must be a non-negative integer'):
        synthetic_image.build_adaptive_grid(pad=-1)
    
    # unhappy path: non-integer padding
    with pytest.raises(ValueError, match='Padding value must be a non-negative integer'):
        synthetic_image.build_adaptive_grid(pad=1.5)


def test_get_image_positions():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)
    
    # check angular positions
    angular_positions = synthetic_image.get_image_positions(pixel=False)
    expected_angular_result = (np.array([ 1.12731457, -0.56841653]), np.array([-1.50967129,  0.57814324]))
    np.testing.assert_allclose(angular_positions[0], expected_angular_result[0], rtol=1e-5)
    np.testing.assert_allclose(angular_positions[1], expected_angular_result[1], rtol=1e-5)

    # check pixel positions
    pixel_positions = synthetic_image.get_image_positions(pixel=True)
    expected_pixel_result = (np.array([33.24831427, 17.83257703]), np.array([ 9.27571558, 28.25584762]))
    np.testing.assert_allclose(pixel_positions[0], expected_pixel_result[0], rtol=1e-5)
    np.testing.assert_allclose(pixel_positions[1], expected_pixel_result[1], rtol=1e-5)


def test_plot():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False,
                                     verbose=False)
    synthetic_image.plot()     
