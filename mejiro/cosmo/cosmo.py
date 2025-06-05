import numpy as np


def stellar_to_main_halo_mass(stellar_mass, z, sample=True):
    """
    Converts stellar mass to main halo mass.

    The model is based on Equation 14 from Lagattuta et al. 2010:
    https://iopscience.iop.org/article/10.1088/0004-637X/716/2/1579.
    Note that it is an empirical relation derived from systems z < 1.

    Parameters
    ----------
    stellar_mass : float
        The stellar mass in solar masses.
    z : float
        The redshift.
    sample : bool, optional
        If True, introduces stochasticity in the scaling factor and exponent
        using truncated normal distributions. If False, uses fixed values
        for `alpha` (51) and `beta` (0.9). Default is True.

    Returns
    -------
    float
        The main halo mass in solar masses.

    Notes
    -----
    - When `sample` is False:
        - The scaling factor (`alpha`) is fixed at 51.
        - The exponent (`beta`) is fixed at 0.9.
    - When `sample` is True:
        - The scaling factor (`alpha`) is sampled from a truncated normal 
          distribution with mean 51 and standard deviation 36, truncated to 
          the range [-1, 1].
        - The exponent (`beta`) is sampled from a truncated normal distribution 
          with mean 0.9 and standard deviation 1.8, truncated to the range [-1, 1].
    """
    from scipy.stats import truncnorm
    
    if not sample:
        return stellar_mass * 51 * (1 + z) ** 0.9
    else:
        rv1 = truncnorm(a=-1, b=1, loc=51, scale=36)
        rv2 = truncnorm(a=-1, b=1, loc=0.9, scale=1.8)
        alpha = rv1.rvs()
        beta = rv2.rvs()
        return stellar_mass * alpha * (1 + z) ** beta
    

def mass_to_einstein_radius(m, d_l, d_s, d_ls):
    return np.sqrt(m / np.power(10, 11.09)) * np.sqrt(d_ls / (d_l * d_s)) 


def einstein_radius_to_mass(einstein_radius, d_l, d_s, d_ls):
    return ((d_l * d_s) / d_ls) * np.power(einstein_radius, 2) * np.power(10, 11.09)


def einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source, cosmo):
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo

    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    return lens_cosmo.sis_theta_E2sigma_v(einstein_radius)


def velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source, cosmo):
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo

    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    return lens_cosmo.sis_sigma_v2theta_E(velocity_dispersion)


def redshift_to_comoving_distance(redshift, cosmo):
    """
    Convert redshift to comoving distance.

    Parameters
    ----------
    redshift : float
        The redshift.
    cosmo : astropy.cosmology.Cosmology
        The Astropy Cosmology instance.

    Returns
    -------
    astropy.units.Quantity
        The comoving distance in gigaparsecs (Gpc).
    """
    import astropy.cosmology.units as cu
    import astropy.units as u

    z = redshift * cu.redshift
    return z.to(u.Gpc, cu.redshift_distance(cosmo, kind='comoving'))
