import pickle

import astropy.cosmology as astropy_cosmo
import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.preset_models import CDM


def get_lens_cosmo(z_lens, z_source):
    return LensCosmo(z_lens=z_lens,
                     z_source=z_source,
                     cosmo=get_default_cosmo())


def cut_mass(realization, log_m_cutoff):
    for i, halo in enumerate(realization.halos):
        if halo.mass < 10 ** log_m_cutoff:
            del realization.halos[i]

    return realization


def generate_CDM_halos(z_lens, z_source, log_m_host=13.3, r_tidal=0.25, cone_opening_angle_arcsec=11,
                       LOS_normalization=0.0):
    realizationCDM = CDM(z_lens, z_source, log_m_host=log_m_host, r_tidal=r_tidal,
                         cone_opening_angle_arcsec=cone_opening_angle_arcsec,
                         LOS_normalization=LOS_normalization)

    # calculate total subhalo mass in solar masses
    total_subhalo_mass = np.sum([halo.mass for halo in realizationCDM.halos])

    return realization_to_lensing_quantities(realizationCDM), total_subhalo_mass


def realization_to_lensing_quantities(realization):
    # set cosmology by initializing pyHalo's Cosmology object, otherwise Colossus throws an error down the line
    astropy_default_cosmo = get_default_cosmo()
    Cosmology(astropy_instance=astropy_default_cosmo)

    # generate lenstronomy objects
    halo_lens_model_list, halo_redshift_list, kwargs_halos, _ = realization.lensing_quantities()

    # for some reason, halo_lens_model_list and kwargs_halos are lists, but halo_redshift_list is an ndarray
    halo_redshift_list = halo_redshift_list.tolist()

    return halo_lens_model_list, halo_redshift_list, kwargs_halos


def unpickle_subhalos(filepath):
    with open(filepath, 'rb') as results_file:
        halo_lens_model_list, halo_redshift_list, kwargs_halos = pickle.load(results_file)

    return halo_lens_model_list, halo_redshift_list, kwargs_halos


def get_default_cosmo():
    return astropy_cosmo.default_cosmology.get()
