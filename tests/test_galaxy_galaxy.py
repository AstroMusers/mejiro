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


def test_get_lens_cosmo():
    strong_lens = SampleGG()
    
    lens_cosmo = strong_lens.get_lens_cosmo()
    assert lens_cosmo.z_lens == 0.2902115249535011
    assert lens_cosmo.z_source == 0.5876899931818929


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
    