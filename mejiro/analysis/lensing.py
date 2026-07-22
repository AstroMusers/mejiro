import re

import numpy as np

from mejiro.utils import util

# Sentinel written to StrongLens.realization when the ~59 MB pyHalo realization
# object is stripped for the lightweight serialization (see strip_realization).
# Value must match the literal LightweightStrongLens attaches on load
# (mejiro/synthetic_image.py); a unit test enforces the agreement.
LIGHTWEIGHT_REALIZATION = '<lightweight>'


def get_alpha(lens_model, kwargs_lens, scene_size, pixel_scale):
    """
    Compute the deflection angle map for a given lens model over a 2D grid. The intended use case is:

    .. code-block:: python

        plt.quiver(*get_alpha(lens_model, kwargs_lens, scene_size, pixel_scale))

    """
    xx, yy = util.build_meshgrid(scene_size, pixel_scale)
    alpha_x, alpha_y = lens_model.alpha(xx.ravel(), yy.ravel(), kwargs_lens)
    return xx, yy, alpha_x, alpha_y


def get_potential(lens_model, kwargs_lens, scene_size, pixel_scale):
    """
    Compute the potential map for a given lens model over a 2D grid. The scene size is calculated from overall scene size and pixel scale to match how scene size is calculated in the SyntheticImage class. This way, the same parameters will yield arrays with the same shapes and can be directly compared.

    Parameters
    ----------
    lens_model : object
        See lenstronomy documentation.
    kwargs_lens : list of dict
        See lenstronomy documentation.
    scene_size : float
        The physical size of the scene in angular units (often, arcseconds).
    pixel_scale : float
        The size of each pixel.

    Returns
    -------
    np.ndarray
        A 2D array containing the potential values evaluated on a grid covering the scene.
    """
    xx, yy = util.build_meshgrid(scene_size, pixel_scale)
    return lens_model.potential(xx.ravel(), yy.ravel(), kwargs_lens).reshape(xx.shape)


def get_kappa(lens_model, kwargs_lens, scene_size, pixel_scale):
    """
    Compute the convergence (kappa) map for a given lens model over a 2D grid. The scene size is calculated from overall scene size and pixel scale to match how scene size is calculated in the SyntheticImage class. This way, the same parameters will yield arrays with the same shapes and can be directly compared.

    Parameters
    ----------
    lens_model : object
        See lenstronomy documentation.
    kwargs_lens : list of dict
        See lenstronomy documentation.
    scene_size : float
        The physical size of the scene in angular units (often, arcseconds).
    pixel_scale : float
        The size of each pixel.

    Returns
    -------
    np.ndarray
        A 2D array containing the convergence (kappa) values evaluated on a grid covering the scene.
    """
    xx, yy = util.build_meshgrid(scene_size, pixel_scale)
    return lens_model.kappa(xx.ravel(), yy.ravel(), kwargs_lens).reshape(xx.shape)


def get_subhalo_mass_function(realization, bins=10):
    """
    Return the subhalo mass function from a pyHalo subhalo realization. The intended use case is quickly plotting the subhalo mass function in the following way:

    .. code-block:: python

        plt.loglog(*get_subhalo_mass_function(realization))

    Parameters
    ----------
    realization : object
        A pyHalo realization.
    bins : int, optional
        Number of bins to use for the mass histogram (default is 10).

    Returns
    -------
    bin_edges : numpy.ndarray
        The edges of the mass bins (excluding the last edge), in the same units as the halo masses.
    hist : numpy.ndarray
        The number of halos in each mass bin.
    """
    halo_masses = [halo.mass for halo in realization.halos if halo.is_subhalo]
    log_mlow = np.floor(np.log10(np.min(halo_masses)))
    log_mhigh = np.ceil(np.log10(np.max(halo_masses)))
    hist, bin_edges = np.histogram(halo_masses, bins=np.logspace(log_mlow, log_mhigh, bins))
    return bin_edges[0:-1], hist


def substructure_flag(realization):
    """Derive the short substructure-model flag (e.g. ``'CDM'``, ``'WDM'``) from a pyHalo realization.

    Reads the mass-function and concentration model class names off the
    realization and extracts the leading all-caps abbreviation.

    Parameters
    ----------
    realization : object
        A pyHalo realization (must expose ``rendering_classes`` and
        ``kwargs_halo_model``).

    Returns
    -------
    str
        The substructure flag, e.g. ``'CDM'`` for ``CDMPowerLaw`` /
        ``ConcentrationLudlow``.
    """
    mf_name = realization.rendering_classes[0]._mass_function_model.__name__
    conc_name = realization.kwargs_halo_model['concentration_model_subhalos'].__class__.__name__
    match = re.match(r'^([A-Z]{2,}?)(?=[A-Z][a-z]|$)', mf_name) \
        or re.search(r'([A-Z]{2,}?)(?=[A-Z][a-z]|$)', conc_name)
    return match.group(1)


def strip_realization(lens):
    """Drop the heavy pyHalo realization object from a StrongLens for lightweight serialization.

    ``StrongLens.add_realization`` bakes the halos into ``kwargs_lens`` /
    ``lens_model_list`` / ``lens_redshift_list`` (all that image generation
    reads) but also retains the full realization object -- ~59 MB / ~100k
    ``Halo`` objects, ~76% of a subhalo'd pickle. This records the substructure
    flag (so downstream analysis keeps the metadata) and then replaces the
    realization with the truthy :data:`LIGHTWEIGHT_REALIZATION` sentinel, so
    ``lens.realization is not None`` still answers correctly (the
    ``has_realization`` flag in the lightweight synthetic image and the
    macromodel branch in ``GalaxyGalaxy.get_image_positions``).

    Parameters
    ----------
    lens : StrongLens
        A lens with a pyHalo realization already attached via
        ``add_realization``. Mutated in place.
    """
    lens.substructure_flag = substructure_flag(lens.realization)
    lens.realization = LIGHTWEIGHT_REALIZATION
