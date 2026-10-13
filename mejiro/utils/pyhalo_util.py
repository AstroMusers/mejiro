"""Small adapters for optional PyHalo substructure objects."""


def make_single_nfw_subhalo(
    lens,
    halo_mass,
    center_x,
    center_y,
    lens_cosmo,
    kwargs_halo_model,
):
    """Create one NFW subhalo for an existing Mejiro lens and PyHalo context."""
    from pyHalo.single_realization import SingleHalo

    return SingleHalo(
        halo_mass=halo_mass,
        x=center_x,
        y=center_y,
        mdef='NFW',
        z=lens.z_lens,
        zlens=lens.z_lens,
        zsource=lens.z_source,
        subhalo_flag=True,
        kwargs_halo_model=kwargs_halo_model,
        astropy_instance=lens.cosmo,
        lens_cosmo=lens_cosmo,
    )