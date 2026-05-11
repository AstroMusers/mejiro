import numpy as np
import pytest
import astropy.units as u
from astropy.cosmology import default_cosmology

from mejiro.cosmo import cosmo


def test_stellar_to_main_halo_mass_deterministic():
    # at z = 0, M_halo = M_star * 51 * (1)^0.9 = 51 * M_star
    assert cosmo.stellar_to_main_halo_mass(1.0, 0.0, sample=False) == pytest.approx(51.0)
    assert cosmo.stellar_to_main_halo_mass(1e10, 0.0, sample=False) == pytest.approx(51e10)

    # at z = 1, M_halo = M_star * 51 * 2^0.9
    expected = 1.0 * 51.0 * (2.0 ** 0.9)
    assert cosmo.stellar_to_main_halo_mass(1.0, 1.0, sample=False) == pytest.approx(expected)

    # linearity in stellar mass
    a = cosmo.stellar_to_main_halo_mass(1.0, 0.5, sample=False)
    b = cosmo.stellar_to_main_halo_mass(7.0, 0.5, sample=False)
    assert b == pytest.approx(7.0 * a)


def test_stellar_to_main_halo_mass_sampled_bounds():
    # with sample=True at z=0, result = M_star * alpha where alpha is drawn from
    # truncnorm(a=-1, b=1, loc=51, scale=36); the truncation puts alpha in [15, 87]
    np.random.seed(0)
    samples = np.array([cosmo.stellar_to_main_halo_mass(1.0, 0.0, sample=True)
                        for _ in range(500)])

    assert np.all(np.isfinite(samples))
    assert np.all(samples >= 15.0)
    assert np.all(samples <= 87.0)
    # symmetric truncated normal mean equals its location parameter (= 51)
    assert np.mean(samples) == pytest.approx(51.0, abs=3.0)


def test_stellar_to_main_halo_mass_sampled_returns_scalar():
    np.random.seed(1)
    result = cosmo.stellar_to_main_halo_mass(1e10, 0.5, sample=True)
    assert np.isscalar(result) or np.ndim(result) == 0
    assert np.isfinite(result)
    assert result > 0


def test_mass_to_einstein_radius():
    # with m = 10^11.09 and unit distances, theta_E = sqrt(1) * sqrt(1) = 1
    m_star = 10 ** 11.09
    assert cosmo.mass_to_einstein_radius(m_star, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    # quadrupling the mass doubles the Einstein radius
    base = cosmo.mass_to_einstein_radius(m_star, 0.5, 1.0, 0.6)
    quad = cosmo.mass_to_einstein_radius(4 * m_star, 0.5, 1.0, 0.6)
    assert quad == pytest.approx(2 * base)


def test_einstein_radius_to_mass():
    # with theta_E = 1 and unit distances, m = 10^11.09
    assert cosmo.einstein_radius_to_mass(1.0, 1.0, 1.0, 1.0) == pytest.approx(10 ** 11.09)

    # squaring the Einstein radius quadruples the inferred mass
    base = cosmo.einstein_radius_to_mass(0.5, 0.4, 1.5, 1.1)
    doubled = cosmo.einstein_radius_to_mass(1.0, 0.4, 1.5, 1.1)
    assert doubled == pytest.approx(4 * base)


@pytest.mark.parametrize("m, d_l, d_s, d_ls", [
    (1e11, 1.0, 1.5, 1.2),
    (5e11, 0.8, 2.0, 1.6),
    (1e12, 1.2, 2.5, 1.9),
])
def test_mass_einstein_radius_roundtrip(m, d_l, d_s, d_ls):
    theta_E = cosmo.mass_to_einstein_radius(m, d_l, d_s, d_ls)
    recovered = cosmo.einstein_radius_to_mass(theta_E, d_l, d_s, d_ls)
    assert recovered == pytest.approx(m, rel=1e-12)


def test_einstein_radius_to_velocity_dispersion():
    astropy_cosmo = default_cosmology.get()
    sigma_v = cosmo.einstein_radius_to_velocity_dispersion(
        einstein_radius=1.0, z_lens=0.3, z_source=1.5, cosmo=astropy_cosmo
    )
    # SIS velocity dispersion for a ~1" Einstein radius at typical redshifts
    # falls in the range of bright elliptical galaxies (~150-400 km/s)
    assert np.isfinite(sigma_v)
    assert 150.0 < sigma_v < 400.0


def test_velocity_dispersion_to_einstein_radius():
    astropy_cosmo = default_cosmology.get()
    theta_E = cosmo.velocity_dispersion_to_einstein_radius(
        velocity_dispersion=220.0, z_lens=0.3, z_source=1.5, cosmo=astropy_cosmo
    )
    # a 220 km/s SIS lens at typical redshifts produces a sub-arcsecond Einstein radius
    assert np.isfinite(theta_E)
    assert 0.3 < theta_E < 2.0


@pytest.mark.parametrize("z_lens, z_source, theta_E", [
    (0.2, 1.0, 0.8),
    (0.5, 2.0, 1.2),
    (0.4, 1.5, 1.5),
])
def test_einstein_radius_velocity_dispersion_roundtrip(z_lens, z_source, theta_E):
    astropy_cosmo = default_cosmology.get()
    sigma_v = cosmo.einstein_radius_to_velocity_dispersion(
        theta_E, z_lens, z_source, astropy_cosmo
    )
    recovered = cosmo.velocity_dispersion_to_einstein_radius(
        sigma_v, z_lens, z_source, astropy_cosmo
    )
    assert recovered == pytest.approx(theta_E, rel=1e-6)


def test_redshift_to_comoving_distance_at_zero():
    astropy_cosmo = default_cosmology.get()
    d = cosmo.redshift_to_comoving_distance(0.0, astropy_cosmo)
    assert d.to(u.Gpc).value == pytest.approx(0.0, abs=1e-12)


def test_redshift_to_comoving_distance_units():
    astropy_cosmo = default_cosmology.get()
    d = cosmo.redshift_to_comoving_distance(1.0, astropy_cosmo)

    assert isinstance(d, u.Quantity)
    assert d.unit.is_equivalent(u.Gpc)


def test_redshift_to_comoving_distance_monotonic():
    astropy_cosmo = default_cosmology.get()
    distances = [cosmo.redshift_to_comoving_distance(z, astropy_cosmo).to(u.Gpc).value
                 for z in [0.1, 0.5, 1.0, 2.0, 3.0]]
    assert np.all(np.diff(distances) > 0)


@pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 2.5])
def test_redshift_to_comoving_distance_matches_astropy(z):
    astropy_cosmo = default_cosmology.get()
    d = cosmo.redshift_to_comoving_distance(z, astropy_cosmo).to(u.Gpc)
    expected = astropy_cosmo.comoving_distance(z).to(u.Gpc)
    assert d.value == pytest.approx(expected.value, rel=1e-6)
