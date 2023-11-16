import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from scipy.stats import norm, truncnorm, uniform


class Simple:

    def __init__(self, num):
        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        lens_model_list = ['SIE', 'SHEAR']
        self.lens_model_class = LensModel(lens_model_list)

        loc = 1.2
        scale = 0.2
        clip_a = 0.5
        clip_b = 2.
        a = (clip_a - loc) / scale
        b = (clip_b - loc) / scale

        self.deflector_theta_e = truncnorm(a=a, b=b, loc=loc, scale=scale).rvs(size=num)
        self.deflector_e1 = norm(loc=0.0, scale=0.1).rvs(size=num)
        self.deflector_e2 = norm(loc=0.0, scale=0.1).rvs(size=num)
        self.deflector_center_x = norm(loc=0.0, scale=0.16).rvs(size=num)
        self.deflector_center_y = norm(loc=0.0, scale=0.16).rvs(size=num)
        self.deflector_gamma1 = norm(loc=0.0, scale=0.05).rvs(size=num)
        self.deflector_gamma2 = norm(loc=0.0, scale=0.05).rvs(size=num)

        self.deflector_headers = ['deflector_theta_E', 'theta_E_class', 'deflector_e1', 'deflector_e2',
                                  'deflector_center_x', 'deflector_center_y', 'deflector_gamma1', 'deflector_gamma2']

        # light model: sersic ellipse profile
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.lens_light_model_class = LightModel(lens_light_model_list)

        self.deflector_light_amp = uniform(loc=100, scale=10).rvs(size=num)
        self.deflector_light_r_sersic = truncnorm(-3, 3, loc=0.5, scale=0.1).rvs(size=num)
        self.deflector_light_n_sersic = truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(size=num)
        self.deflector_light_e1 = norm(loc=0.0, scale=0.1).rvs(size=num)
        self.deflector_light_e2 = norm(loc=0.0, scale=0.1).rvs(size=num)
        self.deflector_light_center_x = self.deflector_center_x
        self.deflector_light_center_y = self.deflector_center_y

        self.deflector_light_headers = ['deflector_light_R_sersic', 'deflector_light_n_sersic', 'deflector_light_e1',
                                        'deflector_light_e2', 'deflector_light_center_x', 'deflector_light_center_y']

        # SOURCE
        # light model: sersic ellipse profile
        source_model_list = ['SERSIC_ELLIPSE']
        self.source_model_class = LightModel(source_model_list)

        self.source_light_amp = uniform(loc=10, scale=1).rvs(size=num)
        self.source_light_r_sersic = truncnorm(-2, 2, loc=0.35, scale=0.05).rvs(size=num)
        self.source_light_n_sersic = truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(size=num)
        self.source_light_e1 = norm(loc=0.0, scale=0.1).rvs(size=num)
        self.source_light_e2 = norm(loc=0.0, scale=0.1).rvs(size=num)
        self.source_light_center_x = norm(loc=0.0, scale=0.16).rvs(size=num)
        self.source_light_center_y = norm(loc=0.0, scale=0.16).rvs(size=num)

        self.source_light_headers = ['source_light_R_sersic', 'source_light_n_sersic', 'source_light_e1',
                                     'source_light_e2', 'source_light_center_x', 'source_light_center_y']

        self.kwargs_model = {
            'lens_model_list': lens_model_list,
            'lens_light_model_list': lens_light_model_list,
            'source_light_model_list': source_model_list
        }

    def get_kwargs_lens(self, i):
        kwargs_spemd = {
            'theta_E': self.deflector_theta_e[i],
            'e1': self.deflector_e1[i],
            'e2': self.deflector_e2[i],
            'center_x': self.deflector_center_x[i],
            'center_y': self.deflector_center_y[i]
        }
        kwargs_shear = {
            'gamma1': self.deflector_gamma1[i],
            'gamma2': self.deflector_gamma2[i]
        }

        return [kwargs_spemd, kwargs_shear]

    def get_kwargs_lens_light(self, i):
        kwargs_sersic_lens = {
            'amp': self.deflector_light_amp[i],
            'R_sersic': self.deflector_light_r_sersic[i],
            'n_sersic': self.deflector_light_n_sersic[i],
            'e1': self.deflector_light_e1[i],
            'e2': self.deflector_light_e2[i],
            'center_x': self.deflector_light_center_x[i],
            'center_y': self.deflector_light_center_y[i]
        }

        return [kwargs_sersic_lens]

    def get_kwargs_source(self, i):
        kwargs_sersic = {
            'amp': self.source_light_amp[i],
            'R_sersic': self.source_light_r_sersic[i],
            'n_sersic': self.source_light_n_sersic[i],
            'e1': self.source_light_e1[i],
            'e2': self.source_light_e2[i],
            'center_x': self.source_light_center_x[i],
            'center_y': self.source_light_center_y[i]
        }

        return [kwargs_sersic]
