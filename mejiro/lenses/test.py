import os
from glob import glob

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util

from mejiro.helpers import color
from mejiro.lenses import lens_util
from mejiro.lenses.lens import Lens
from mejiro.utils import util


def get_sample(pickle_dir, pandeia_dir, index):
    files = glob(pickle_dir + f'/lens_dict_{str(index).zfill(8)}_*')

    f106 = [util.unpickle(i) for i in files if 'f106' in i][0]
    f129 = [util.unpickle(i) for i in files if 'f129' in i][0]
    # f158 = [util.unpickle(i) for i in files if 'f158' in i][0]
    f184 = [util.unpickle(i) for i in files if 'f184' in i][0]

    rgb_model = color.get_rgb(f106['model'], f129['model'], f184['model'], minimum=None, stretch=3, Q=8)

    image_path = os.path.join(pandeia_dir, f'pandeia_color_{str(index).zfill(8)}.npy')
    rgb_image = np.load(image_path)

    return f106, rgb_image, rgb_model


class SampleSkyPyLens(Lens):
    def __init__(self):
        # define redshifts
        self.z_lens = 0.643971
        self.z_source = 1.627633

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['SIE', 'SHEAR']
        self.lens_redshift_list = [self.z_lens] * len(self.lens_model_list)
        kwargs_mass = {
            # 'sigma_v': 250,  # velocity dispersion in units km/s
            'theta_E': 0.975908,
            'center_x': 0.042010,
            'center_y': 0.038204,
            'e1': 0.1,
            'e2': 0
        }
        kwargs_shear = {
            'gamma1': 0.1,
            'gamma2': -0.05
        }
        self.kwargs_lens = [kwargs_mass, kwargs_shear]

        # light model: sersic ellipse profile
        self.lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_sersic_lens = {
            'magnitude': 20.934556,
            'R_sersic': 0.6,
            'n_sersic': 2,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.042010,
            'center_y': 0.038204
        }
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        self.source_model_list = ['SERSIC_ELLIPSE']
        self.source_redshift_list = [self.z_source]
        kwargs_sersic = {
            'magnitude': 23.902054,
            'R_sersic': 0.1,
            'n_sersic': 1,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.237250,
            'center_y': -0.090416
        }
        self.kwargs_source = [kwargs_sersic]

        # set kwargs_params
        self.kwargs_params = lens_util.set_kwargs_params(self.kwargs_lens, self.kwargs_lens_light, self.kwargs_source)

        # set kwargs_model
        self.kwargs_model = lens_util.set_kwargs_model(self.lens_model_list, self.lens_light_model_list,
                                                       self.source_model_list)
        self.kwargs_model['lens_redshift_list'] = self.lens_redshift_list
        self.kwargs_model['source_redshift_list'] = self.source_redshift_list
        self.kwargs_model['z_source'] = self.z_source

        super().__init__(kwargs_model=self.kwargs_model, kwargs_params=self.kwargs_params, band='f106')


class TestLens(Lens):
    def __init__(self):
        # define redshifts
        self.z_lens = 0.5
        self.z_source = 1.5

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['SIE', 'SHEAR']
        self.lens_redshift_list = [self.z_lens] * len(self.lens_model_list)
        kwargs_mass = {
            # 'sigma_v': 250,  # velocity dispersion in units km/s
            'theta_E': 1.,
            'center_x': 0,
            'center_y': 0,
            'e1': 0.1,
            'e2': 0
        }
        kwargs_shear = {
            'gamma1': 0.1,
            'gamma2': -0.05
        }
        self.kwargs_lens = [kwargs_mass, kwargs_shear]

        # light model: sersic ellipse profile
        self.lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_sersic_lens = {
            'magnitude': 22,
            'R_sersic': 0.6,
            'n_sersic': 2,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0,
            'center_y': 0
        }
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        self.source_model_list = ['SERSIC_ELLIPSE']
        self.source_redshift_list = [self.z_source]
        kwargs_sersic = {
            'magnitude': 26,
            'R_sersic': 0.1,
            'n_sersic': 1,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.1,
            'center_y': -0.1
        }
        self.kwargs_source = [kwargs_sersic]

        # set kwargs_params
        self.kwargs_params = lens_util.set_kwargs_params(self.kwargs_lens, self.kwargs_lens_light, self.kwargs_source)

        # set kwargs_model
        self.kwargs_model = lens_util.set_kwargs_model(self.lens_model_list, self.lens_light_model_list,
                                                       self.source_model_list)
        self.kwargs_model['lens_redshift_list'] = self.lens_redshift_list
        self.kwargs_model['source_redshift_list'] = self.source_redshift_list
        self.kwargs_model['z_source'] = self.z_source

        super().__init__(kwargs_model=self.kwargs_model, kwargs_params=self.kwargs_params, band='f106')


class TutorialLens(Lens):
    def __init__(self):
        # define redshifts
        self.z_lens = 0.5
        self.z_source = 1.5

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['EPL', 'SHEAR']
        self.lens_redshift_list = [self.z_lens] * len(self.lens_model_list)
        kwargs_spep = {'theta_E': 1.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 2., 'center_x': 0.1, 'center_y': 0}
        kwargs_shear = {'gamma1': -0.01, 'gamma2': .03}
        self.kwargs_lens = [kwargs_spep, kwargs_shear]

        # light model: sersic ellipse profile
        self.lens_light_model_list = ['SERSIC_ELLIPSE']
        e1, e2 = param_util.phi_q2_ellipticity(phi=0.5, q=0.7)
        kwargs_sersic_lens = {'amp': 1000, 'R_sersic': 0.1, 'n_sersic': 2.5, 'e1': e1, 'e2': e2, 'center_x': 0.1,
                              'center_y': 0}
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        self.source_model_list = ['SERSIC']
        self.source_redshift_list = [self.z_source]
        theta_ra, theta_dec = 1., .5
        self.lens_model_class = LensModel(self.lens_model_list)
        beta_ra, beta_dec = self.lens_model_class.ray_shooting(theta_ra, theta_dec, self.kwargs_lens)
        kwargs_sersic = {'amp': 100, 'R_sersic': 0.1, 'n_sersic': 1.5, 'center_x': beta_ra, 'center_y': beta_dec}
        self.kwargs_source = [kwargs_sersic]

        # set kwargs_params
        self.kwargs_params = lens_util.set_kwargs_params(self.kwargs_lens, self.kwargs_lens_light, self.kwargs_source)

        # set kwargs_model
        self.kwargs_model = lens_util.set_kwargs_model(self.lens_model_list, self.lens_light_model_list,
                                                       self.source_model_list)
        self.kwargs_model['lens_redshift_list'] = self.lens_redshift_list
        self.kwargs_model['source_redshift_list'] = self.source_redshift_list
        self.kwargs_model['z_source'] = self.z_source

        super().__init__(kwargs_model=self.kwargs_model, kwargs_params=self.kwargs_params, band='f106')
