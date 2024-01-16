import numpy as np
from scipy.stats import norm, truncnorm

from mejiro.lenses import lens_util
from mejiro.lenses.stronglens import StrongLens


class RandomStrongLens(StrongLens):
    def __init__(self, z_lens, z_source, theta_e, lens_x, lens_y, source_x, source_y, mag_lens, mag_source):
        # define redshifts
        self.z_lens = z_lens
        self.z_source = z_source

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['SIE', 'SHEAR']
        self.lens_redshift_list = [self.z_lens] * len(self.lens_model_list)
        kwargs_sie = {
            # 'sigma_v': sigma_v,  # velocity dispersion in units km/s
            'theta_E': theta_e,
            'center_x': lens_x,
            'center_y': lens_y,
            'e1': norm(loc=0.0, scale=0.1).rvs(),
            'e2': norm(loc=0.0, scale=0.1).rvs()
        }
        kwargs_shear = {
            'gamma1': norm(loc=0.0, scale=0.05).rvs(),
            'gamma2': norm(loc=0.0, scale=0.05).rvs()
        }
        self.kwargs_lens = [kwargs_sie, kwargs_shear]

        # light model: sersic ellipse profile
        self.lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_sersic_lens = {
            'magnitude': mag_lens,
            'R_sersic': truncnorm(-2, 2, loc=0.35, scale=0.05).rvs(),
            'n_sersic': truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(),
            'e1': norm(loc=0.0, scale=0.1).rvs(),
            'e2': norm(loc=0.0, scale=0.1).rvs(),
            'center_x': lens_x,
            'center_y': lens_y
        }
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        self.source_model_list = ['SERSIC_ELLIPSE']
        self.source_redshift_list = [self.z_source]
        kwargs_sersic_source = {
            'magnitude': mag_source,
            'R_sersic': truncnorm(-2, 2, loc=0.35, scale=0.05).rvs(),
            'n_sersic': truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(),
            'e1': norm(loc=0.0, scale=0.1).rvs(),
            'e2': norm(loc=0.0, scale=0.1).rvs(),
            'center_x': source_x,
            'center_y': source_y
        }
        self.kwargs_source = [kwargs_sersic_source]

        # set kwargs_params
        self.kwargs_params = lens_util.set_kwargs_params(self.kwargs_lens, self.kwargs_lens_light, self.kwargs_source)

        # set kwargs_model
        self.kwargs_model = lens_util.set_kwargs_model(self.lens_model_list, self.lens_light_model_list,
                                                       self.source_model_list)
        self.kwargs_model['lens_redshift_list'] = self.lens_redshift_list
        self.kwargs_model['source_redshift_list'] = self.source_redshift_list
        self.kwargs_model['z_source'] = self.z_source

        super().__init__(kwargs_model=self.kwargs_model, kwargs_params=self.kwargs_params)
