from pyHalo.preset_models import CDM


def generate_CDM_halos(z_lens, z_source):
    realizationCDM = CDM(z_lens, z_source, cone_opening_angle_arcsec=10, LOS_normalization=0.0)

    halo_lens_model_list, halo_redshift_list, kwargs_halos, _ = realizationCDM.lensing_quantities()

    # for some reason, halo_lens_model_list and kwargs_halos are lists, but halo_redshift_list is an ndarray
    halo_redshift_list = halo_redshift_list.tolist()

    return halo_lens_model_list, halo_redshift_list, kwargs_halos
