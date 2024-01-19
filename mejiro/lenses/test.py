from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util

from mejiro.lenses import lens_util
from mejiro.lenses.strong_lens import StrongLens


class SampleStrongLens(StrongLens):
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
            'magnitude': 1,  # TODO test
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
        kwargs_sersic = {  # TODO test, no magnitude attribute
            'R_sersic': 0.1,
            'n_sersic': 1,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.237250,
            'center_y': -0.090416
        }
        self.kwargs_source = [kwargs_sersic]

        lens_mags = {
            'F106': 20.47734355,
            'F129': 20.1486012,
            'F158': 19.94610498,
            'F184': 19.85683648
        }
        source_mags = {
            'F106': 22.18633408,
            'F129': 21.98034088,
            'F158': 21.77380172,
            'F184': 21.649701
        }

        # set kwargs_params, just for the StrongLens constructor
        kwargs_params = {'kwargs_lens': self.kwargs_lens, 
                         'kwargs_source': self.kwargs_source,
                         'kwargs_lens_light': self.kwargs_lens_light}

        # set kwargs_model
        self.kwargs_model = lens_util.set_kwargs_model(self.lens_model_list, self.lens_light_model_list,
                                                       self.source_model_list)
        self.kwargs_model['lens_redshift_list'] = self.lens_redshift_list
        self.kwargs_model['source_redshift_list'] = self.source_redshift_list
        self.kwargs_model['z_source'] = self.z_source

        super().__init__(kwargs_model=self.kwargs_model, kwargs_params=kwargs_params, lens_mags=lens_mags,
                         source_mags=source_mags)


# TODO fix
class TestStrongLens(StrongLens):
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


# TODO fix
class TutorialStrongLens(StrongLens):
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
