import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from slsim import lens as slsim_lens

from mejiro.galaxy_galaxy import GalaxyGalaxy


def slsim_lens_to_mejiro(slsim_lens, bands, cosmo):
    kwargs_model, kwargs_params = slsim_lens.lenstronomy_kwargs(band=bands[0])

    z_lens, z_source = slsim_lens.deflector_redshift, slsim_lens.source_redshift_list[0]  # TODO confirm that first element of source_redshift_list will give the appropriate source. for galaxy-galaxy lensing, this will be the case, so this is fine for now.
    kwargs_lens = kwargs_params['kwargs_lens']

    # add additional necessary key/value pairs to kwargs_model
    kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
    kwargs_model['source_redshift_list'] = [z_source]
    kwargs_model['cosmo'] = cosmo
    kwargs_model['z_source'] = z_source
    kwargs_model['z_source_convention'] = 6

    # populate magnitudes dictionary
    lens_mags, source_mags, lensed_source_mags = {}, {}, {}
    for band in bands:
        lens_mags[band] = slsim_lens.deflector_magnitude(band)
        source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=False)[0]  # TODO first element
        lensed_source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=True)[0]  # TODO confirm that first element of source_redshift_list will give the appropriate source. for galaxy-galaxy lensing, this will be the case, so this is fine for now.
    magnitudes = {
        'lens': lens_mags,
        'source': source_mags,
        'lensed_source': lensed_source_mags,
    }

    # populate physical parameters dictionary
    physical_params = {
        'lens_stellar_mass': slsim_lens.deflector_stellar_mass(),
        'lens_velocity_dispersion': slsim_lens.deflector_velocity_dispersion(),
        'magnification': slsim_lens.extended_source_magnification(),
    }

    return GalaxyGalaxy(name=None,
                        coords=None,
                        kwargs_model=kwargs_model,
                        kwargs_params=kwargs_params,
                        magnitudes=magnitudes,
                        physical_params=physical_params,)


def write_lens_pop_to_csv(output_path, gg_lenses, detectable_snr_list, bands, verbose=False):
    dictparaggln = {}
    dictparaggln['Candidate'] = {}
    listnamepara = ['velodisp', 'massstel', 'angleins', 'redssour', 'redslens', 'magnsour', 'snr', 'numbimag',
                    'maxmdistimag']  # 'xposlens', 'yposlens', 'xpossour', 'ypossour',
    for nameband in bands:
        listnamepara += ['magtlens%s' % nameband]
        listnamepara += ['magtsour%s' % nameband]
        listnamepara += ['magtsourMagnified%s' % nameband]

    for namepara in listnamepara:
        dictparaggln['Candidate'][namepara] = np.empty(len(gg_lenses))

    df = pd.DataFrame(columns=listnamepara)

    for i, (gg_lens, snr) in tqdm(enumerate(zip(gg_lenses, detectable_snr_list)), total=len(gg_lenses),
                                  disable=not verbose):
        dict = {
            'velodisp': gg_lens.deflector_velocity_dispersion(),
            'massstel': gg_lens.deflector_stellar_mass() * 1e-12,
            'angleins': gg_lens.einstein_radius,
            'redssour': gg_lens.source_redshift_list[0],  # TODO confirm that first element of source_redshift_list will give the appropriate source. for galaxy-galaxy lensing, this will be the case, so this is fine for now.
            'redslens': gg_lens.deflector_redshift,
            'magnsour': gg_lens.extended_source_magnification(),
            'snr': snr
        }

        posiimag = gg_lens.point_source_image_positions()
        dict['numbimag'] = len(posiimag[0])
        dict['maxmdistimag'] = 0  # slsim_lens.image_separation_from_positions(posiimag) TODO TEMP

        # TODO ypossour was throwing index 1 out of bounds. but I also don't need this info (for now) so maybe just delete
        # posilens = gg_lens.deflector_position
        # posisour = gg_lens.extended_source_image_positions()[0]
        # dict['xposlens'] = posilens[0]
        # dict['yposlens'] = posilens[1]
        # dict['xpossour'] = posisour[0]
        # dict['ypossour'] = posisour[1]

        for nameband in bands:
            dict['magtlens%s' % nameband] = gg_lens.deflector_magnitude(band=nameband)
            dict['magtsour%s' % nameband] = gg_lens.extended_source_magnitude(band=nameband, lensed=False)
            dict['magtsourMagnified%s' % nameband] = gg_lens.extended_source_magnitude(band=nameband, lensed=True)

        df.loc[i] = pd.Series(dict)

    if verbose: print('Writing to %s..' % output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    df.to_csv(output_path, index=False)
