# import numpy as np
# import yaml

# from astropy.cosmology import default_cosmology
# from pyHalo.preset_models import CDM

# from mejiro.lenses.test import SampleStrongLens

# with open("test_config.yaml") as f:
#     cfg = yaml.load(f, Loader=yaml.FullLoader)

# # TODO generate some default CDM subhalo population for the tests

# # TODO get a sample lens to get the params below
# lens = SampleStrongLens()
# z_lens = lens.z_lens
# z_source = lens.z_source
# log_m_host = np.log10(lens.main_halo_mass)

# r_tidal = cfg['pipeline']['r_tidal']
# sigma_sub = cfg['pipeline']['sigma_sub']
# subhalo_cone = cfg['pipeline']['subhalo_cone']
# los_normalization = cfg['pipeline']['los_normalization']

# cdm_realization = CDM(z_lens,
#                         z_source,
#                         sigma_sub=sigma_sub,
#                         log_mlow=6.,
#                         log_mhigh=10.,
#                         log_m_host=log_m_host,
#                         r_tidal=r_tidal,
#                         cone_opening_angle_arcsec=subhalo_cone,
#                         LOS_normalization=los_normalization)

# def test_subhalo_cosmo():
#     # TODO expect a discrepancy here because e.g. T_cmb isn't being set in the CDM object, so likely not used by pyhalo and thus a non-issue, but need to confirm this
#     assert cdm_realization.cosmo == default_cosmology.get()
