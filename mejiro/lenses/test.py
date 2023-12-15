from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import param_util

from mejiro.lenses.lens import Lens


class SampleSkyPyLens(Lens):
    def __init__(self):
        super().__init__(z_lens=None,
                         z_source=None,
                         theta_e=None,
                         lens_x=None,
                         lens_y=None,
                         source_x=None,
                         source_y=None,
                         mag_lens=None,
                         mag_source=None)

        # define redshifts and cosmology
        self.z_lens = 0.643971
        self.z_source = 1.627633

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['SIE', 'SHEAR']
        self.lens_redshift_list = [self.z_lens, self.z_lens]
        self.lens_model_class = LensModel(self.lens_model_list)
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
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
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
        self.source_model_class = LightModel(self.source_model_list)
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

        # set up model
        self.kwargs_model = {
            'lens_model_list': self.lens_model_list,
            'lens_redshift_list': self.lens_redshift_list,
            'lens_light_model_list': self.lens_light_model_list,
            'source_light_model_list': self.source_model_list,
            'source_redshift_list': self.source_redshift_list,
            'cosmo': self.cosmo,
            'z_source': self.z_source,
            'z_source_convention': 4
        }

        self._set_amp_light_kwargs()


class TestLens(Lens):
    def __init__(self):
        super().__init__(z_lens=None,
                         z_source=None,
                         theta_e=None,
                         lens_x=None,
                         lens_y=None,
                         source_x=None,
                         source_y=None,
                         mag_lens=None,
                         mag_source=None)

        # define redshifts
        self.z_lens = 0.5
        self.z_source = 1.5

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['SIE', 'SHEAR']
        self.lens_redshift_list = [self.z_lens, self.z_lens]
        self.lens_model_class = LensModel(self.lens_model_list)
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
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
        kwargs_sersic_lens = {
            'magnitude': 22,
            'R_sersic': 0.6,
            'n_sersic': 2,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.05,
            'center_y': 0
        }
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        self.source_model_list = ['SERSIC_ELLIPSE']
        self.source_redshift_list = [self.z_source]
        self.source_model_class = LightModel(self.source_model_list)
        kwargs_sersic = {
            'magnitude': 26,
            'R_sersic': 0.1,
            'n_sersic': 1,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.1,
            'center_y': 0
        }
        self.kwargs_source = [kwargs_sersic]

        # set up model
        self.kwargs_model = {
            'lens_model_list': self.lens_model_list,
            'lens_redshift_list': self.lens_redshift_list,
            'lens_light_model_list': self.lens_light_model_list,
            'source_light_model_list': self.source_model_list,
            'source_redshift_list': self.source_redshift_list,
            'cosmo': self.cosmo,
            'z_source': self.z_source,
            'z_source_convention': 4
        }

        self._set_amp_light_kwargs()


class TutorialLens(Lens):
    def __init__(self):
        super().__init__(z_lens=None,
                         z_source=None,
                         theta_e=None,
                         lens_x=None,
                         lens_y=None,
                         source_x=None,
                         source_y=None,
                         mag_lens=None,
                         mag_source=None)

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        lens_model_list = ['EPL', 'SHEAR']
        self.lens_model_class = LensModel(lens_model_list)
        kwargs_spep = {'theta_E': 1.1, 'e1': 0.1, 'e2': 0.1, 'gamma': 2., 'center_x': 0.1, 'center_y': 0}
        kwargs_shear = {'gamma1': -0.01, 'gamma2': .03}
        self.kwargs_lens = [kwargs_spep, kwargs_shear]

        # light model: sersic ellipse profile
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.lens_light_model_class = LightModel(lens_light_model_list)
        e1, e2 = param_util.phi_q2_ellipticity(phi=0.5, q=0.7)
        kwargs_sersic_lens = {'amp': 1000, 'R_sersic': 0.1, 'n_sersic': 2.5, 'e1': e1, 'e2': e2, 'center_x': 0.1,
                              'center_y': 0}
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        source_model_list = ['SERSIC']
        self.source_model_class = LightModel(source_model_list)
        theta_ra, theta_dec = 1., .5
        beta_ra, beta_dec = self.lens_model_class.ray_shooting(theta_ra, theta_dec, self.kwargs_lens)
        kwargs_sersic = {'amp': 100, 'R_sersic': 0.1, 'n_sersic': 1.5, 'center_x': beta_ra, 'center_y': beta_dec}
        self.kwargs_source = [kwargs_sersic]

        # set up model
        self.kwargs_model = {
            'lens_model_list': lens_model_list,
            'lens_light_model_list': lens_light_model_list,
            'source_light_model_list': source_model_list
        }

        self._set_amp_light_kwargs()
