import astropy.cosmology
from pytest import approx

from mejiro.lenses import strong_lens
from mejiro.lenses.test import SampleStrongLens


def test_init():
    sample_lens = SampleStrongLens()

    # check some attributes
    assert sample_lens.z_lens == 0.643971
    assert sample_lens.z_source == 1.627633

    # TODO check kwargs dicts

    # TODO verify model set up


# TODO finish
def test_get_array():
    sample_lens = SampleStrongLens()

    num_pix = 45
    side = 4.95
    band = 'F184'
    kwargs_psf = {'psf_type': 'NONE'}

    array = sample_lens.get_array(num_pix, side, band, kwargs_psf)

    assert array.shape == (num_pix, num_pix)


def test_velocity_dispersion_to_mass():
    velocity_dispersion = 255.25047210330000
    mass = 0.03243293350855620e12
    calculated_mass = strong_lens.velocity_dispersion_to_mass(velocity_dispersion)
    assert mass == approx(calculated_mass, rel=1e-6)

    velocity_dispersion = 303.6393186131100
    mass = 0.39009390627835700e12
    calculated_mass = strong_lens.velocity_dispersion_to_mass(velocity_dispersion)
    assert mass == approx(calculated_mass, rel=1e-6)


def test_mass_to_velocity_dispersion():
    mass = 0.03243293350855620e12
    velocity_dispersion = 255.25047210330000
    calculated_velocity_dispersion = strong_lens.mass_to_velocity_dispersion(mass)
    assert velocity_dispersion == approx(calculated_velocity_dispersion, rel=1e-6)

    mass = 0.39009390627835700e12
    velocity_dispersion = 303.6393186131100
    calculated_velocity_dispersion = strong_lens.mass_to_velocity_dispersion(mass)
    assert velocity_dispersion == approx(calculated_velocity_dispersion, rel=1e-6)


def test_einstein_radius_to_velocity_dispersion():
    cosmo = astropy.cosmology.default_cosmology.get()

    einstein_radius = 1.4811764093086500
    z_lens = 0.2478444815301060
    z_source = 1.7249902383698100
    velocity_dispersion = 255.25047210330000
    test_velocity_dispersion = strong_lens.einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source,
                                                                                  cosmo)
    assert velocity_dispersion == approx(test_velocity_dispersion, rel=1e-6)

    einstein_radius = 1.9724261658572900
    z_lens = 0.11804160736612800
    z_source = 1.0785714350247300
    velocity_dispersion = 282.2882717656310
    test_velocity_dispersion = strong_lens.einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source,
                                                                                  cosmo)
    assert velocity_dispersion == approx(test_velocity_dispersion, rel=1e-6)


def test_velocity_dispersion_to_einstein_radius():
    cosmo = astropy.cosmology.default_cosmology.get()

    velocity_dispersion = 255.25047210330000
    z_lens = 0.2478444815301060
    z_source = 1.7249902383698100
    einstein_radius = 1.4811764093086500
    test_einstein_radius = strong_lens.velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source,
                                                                              cosmo)
    assert einstein_radius == approx(test_einstein_radius, rel=1e-6)

    velocity_dispersion = 282.2882717656310
    z_lens = 0.11804160736612800
    z_source = 1.0785714350247300
    einstein_radius = 1.9724261658572900
    test_einstein_radius = strong_lens.velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source,
                                                                              cosmo)
    assert einstein_radius == approx(test_einstein_radius, rel=1e-6)


def test_mass_to_einstein_radius():
    cosmo = astropy.cosmology.default_cosmology.get()

    mass_1 = 0.03243293350855620e12
    z_lens_1 = 0.2478444815301060
    z_source_1 = 1.7249902383698100
    einstein_radius_1 = 1.4811764093086500
    calculated_einstein_radius_1 = strong_lens.mass_to_einstein_radius(mass_1, z_lens_1, z_source_1, cosmo)
    assert einstein_radius_1 == approx(calculated_einstein_radius_1, rel=1e-6)

    mass_2 = 0.17994825665559600e12
    z_lens_2 = 1.361153793876400
    z_source_2 = 4.8233821168689500
    einstein_radius_2 = 1.038757305432830
    calculated_einstein_radius_2 = strong_lens.mass_to_einstein_radius(mass_2, z_lens_2, z_source_2, cosmo)
    assert einstein_radius_2 == approx(calculated_einstein_radius_2, rel=1e-6)
