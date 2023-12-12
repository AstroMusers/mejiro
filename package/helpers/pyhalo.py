from pyHalo.preset_models import CDM
from pyHalo.Cosmology.cosmology import Cosmology


def generate_CDM_halos(z_lens, z_source, cone_opening_angle_arcsec=11, LOS_normalization=0.0):
    realizationCDM = CDM(z_lens, z_source, cone_opening_angle_arcsec=cone_opening_angle_arcsec, LOS_normalization=LOS_normalization)

    return realization_to_lensing_quantities(realizationCDM)


def realization_to_lensing_quantities(realization):
    # set cosmology, otherwise Colossus throws an error down the line
    # initializing pyHalo's Cosmology object like this uses all defaults which are used across this pipeline
    cosmo = Cosmology()

    halo_lens_model_list, halo_redshift_list, kwargs_halos, _ = realization.lensing_quantities()

    # for some reason, halo_lens_model_list and kwargs_halos are lists, but halo_redshift_list is an ndarray
    halo_redshift_list = halo_redshift_list.tolist()

    return halo_lens_model_list, halo_redshift_list, kwargs_halos
