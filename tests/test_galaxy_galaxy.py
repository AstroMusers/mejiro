import pytest
import numpy as np
from unittest.mock import MagicMock
from astropy.cosmology import default_cosmology

from mejiro.galaxy_galaxy import GalaxyGalaxy, SampleGG


def test_init():
    strong_lens = SampleGG()
    
    # check inheritance
    assert strong_lens.name == 'SampleGG'
    assert strong_lens.coords is None
    assert isinstance(strong_lens.kwargs_model, dict)
    assert isinstance(strong_lens.kwargs_params, dict)

    # check GalaxyGalaxy __init__()
    assert strong_lens.z_source == 0.5876899931818929
    assert strong_lens.z_lens == 0.2902115249535011
    assert strong_lens.cosmo == default_cosmology.get()
    assert strong_lens.lens_cosmo is None


def test_property():
    strong_lens = SampleGG()

    # check property
    assert strong_lens.kwargs_lens == [
        {
            'center_x': -0.007876281728887604,
            'center_y': 0.010633393703246008,
            'e1': 0.004858808997848661,
            'e2': 0.0075210751726143355,
            'theta_E': 1.168082477232392
        },
        {
            'dec_0': 0,
            'gamma1': -0.03648819840013156,
            'gamma2': -0.06511863424492038,
            'ra_0': 0
        },
        {
            'dec_0': 0,
            'kappa': 0.06020941823541971,
            'ra_0': 0
        }
    ]


def test_get_lens_cosmo():
    strong_lens = SampleGG()
    
    lens_cosmo = strong_lens.get_lens_cosmo()
    assert lens_cosmo.z_lens == 0.2902115249535011
    assert lens_cosmo.z_source == 0.5876899931818929


def test_get_image_positions():
    strong_lens = SampleGG()
    
    image_positions = strong_lens.get_image_positions()
    
    expected_positions = ([1.12731457, -0.56841653], [-1.50967129,  0.57814324])
    
    assert len(image_positions) == len(expected_positions)
    for pos, expected_pos in zip(image_positions, expected_positions):
        assert pos == pytest.approx(expected_pos, rel=1e-5)


def test_from_slsim_source_images():
    """from_slsim() should populate source_images with band-specific arrays for catalog sources."""
    bands = ['F106', 'F158']
    images = {'F106': np.ones((10, 10)) * 1.0, 'F158': np.ones((10, 10)) * 2.0}

    def fake_lenstronomy_kwargs(band):
        kwargs_model = {
            'lens_light_model_list': ['SERSIC_ELLIPSE'],
            'lens_model_list': ['SIE'],
            'source_light_model_list': ['INTERPOL'],
            'point_source_model_list': [],
        }
        kwargs_params = {
            'kwargs_lens': [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0}],
            'kwargs_lens_light': [{'magnitude': 20.0, 'R_sersic': 1.0, 'n_sersic': 4.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0}],
            'kwargs_source': [{'magnitude': 22.0, 'image': images[band], 'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': 0.03}],
            'kwargs_ps': [],
        }
        return kwargs_model, kwargs_params

    slsim_gglens = MagicMock()
    slsim_gglens.source_number = 1
    slsim_gglens.cosmo = default_cosmology.get()
    slsim_gglens.deflector_redshift = 0.4
    slsim_gglens.source_redshift_list = [1.0]
    slsim_gglens.deflector._deflector._deflector_dict = {'mag_F106': 20.0, 'mag_F158': 19.0}
    slsim_gglens.lenstronomy_kwargs.side_effect = fake_lenstronomy_kwargs
    slsim_gglens.deflector_magnitude.return_value = 20.0
    slsim_gglens.extended_source_magnitude.return_value = [22.0]
    slsim_gglens.deflector_stellar_mass.return_value = 1e11
    slsim_gglens.deflector_velocity_dispersion.return_value = 200.0
    slsim_gglens.extended_source_magnification = [5.0]
    slsim_gglens.einstein_radius = [1.0]
    slsim_gglens.deflector.deflector_type = 'EPL'

    gg = GalaxyGalaxy.from_slsim(slsim_gglens, name='test', bands=bands)

    assert 'source_images' in gg.kwargs_params
    for band in bands:
        assert band in gg.kwargs_params['source_images']
        np.testing.assert_array_equal(gg.kwargs_params['source_images'][band], images[band])
