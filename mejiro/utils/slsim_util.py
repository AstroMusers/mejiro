


def slsim_lens_to_mejiro(slsim_lens, bands, cosmo, snr=None, uid=None, z_source_convention=6, sca=None):
    kwargs_model, kwargs_params = slsim_lens.lenstronomy_kwargs(band=bands[0])

    lens_mags, source_mags, lensed_source_mags = {}, {}, {}
    for band in bands:
        lens_mags[band] = slsim_lens.deflector_magnitude(band)
        source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=False)[0]  # TODO first element
        lensed_source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=True)[0]  # TODO first element

    z_lens, z_source = slsim_lens.deflector_redshift, slsim_lens.source_redshift_list[0]  # TODO confirm that first element of source_redshift_list will give the appropriate source. for galaxy-galaxy lensing, this will be the case, so this is fine for now.
    kwargs_lens = kwargs_params['kwargs_lens']

    # add additional necessary key/value pairs to kwargs_model
    kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
    kwargs_model['source_redshift_list'] = [z_source]
    kwargs_model['cosmo'] = cosmo
    kwargs_model['z_source'] = z_source
    kwargs_model['z_source_convention'] = z_source_convention

    # from pprint import pprint
    # pprint(f'{kwargs_model=}')
    # pprint(f'{kwargs_params=}')
    # pprint(f'{lens_mags=}')
    # pprint(f'{source_mags=}')
    # pprint(f'{lensed_source_mags=}')

    return StrongLens(kwargs_model=kwargs_model,
                      kwargs_params=kwargs_params,
                      lens_mags=lens_mags,
                      source_mags=source_mags,
                      lensed_source_mags=lensed_source_mags,
                      lens_stellar_mass=slsim_lens.deflector_stellar_mass(),
                      lens_vel_disp=slsim_lens.deflector_velocity_dispersion(),
                      magnification=slsim_lens.extended_source_magnification(),
                      snr=snr,
                      uid=uid,
                      sca=sca)
