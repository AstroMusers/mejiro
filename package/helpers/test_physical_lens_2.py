from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from package.helpers.lens import Lens


class TestPhysicalLens2(Lens):
    def __init__(self):
        # define redshifts and cosmology
        self.z_lens = 0.3
        self.z_source = 1.1
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        # lens_cosmo = LensCosmo(z_lens=self.z_lens, z_source=self.z_source, cosmo=cosmo)

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        lens_model_list = ['SIE', 'SHEAR']
        lens_redshift_list = [self.z_lens, self.z_lens]
        self.lens_model_class = LensModel(lens_model_list)
        kwargs_mass = {
            'sigma_v': 280,  # velocity dispersion in units km/s
            'center_x': -0.1,
            'center_y': 0.1,
            'e1': 0.1,
            'e2': 0
        }
        kwargs_shear = {
            'gamma1': 0.1,
            'gamma2': -0.05
        }
        self.kwargs_lens = [kwargs_mass, kwargs_shear]

        # light model: sersic ellipse profile
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.lens_light_model_class = LightModel(lens_light_model_list)
        kwargs_sersic_lens = {
            'magnitude': 22,
            'R_sersic': 0.6,
            'n_sersic': 2,
            'e1': 0.4,
            'e2': -0.2,
            'center_x': -0.1,
            'center_y': 0.1
        }
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        source_model_list = ['SERSIC_ELLIPSE']
        source_redshift_list = [self.z_source]
        self.source_model_class = LightModel(source_model_list)
        kwargs_sersic = {
            'magnitude': 25,
            'R_sersic': 0.3,
            'n_sersic': 3,
            'e1': -0.1,
            'e2': 0.1,
            'center_x': 0.1,
            'center_y': -0.1
        }
        self.kwargs_source = [kwargs_sersic]

        # set up model
        self.kwargs_model = {
            'lens_model_list': lens_model_list,
            'lens_redshift_list': lens_redshift_list,
            'lens_light_model_list': lens_light_model_list,
            'source_light_model_list': source_model_list,
            'source_redshift_list': source_redshift_list,
            'cosmo': cosmo,
            'z_source': self.z_source,
            'z_source_convention': 3,
            # source redshift to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
        }

        self.delta_pix, self.num_pix = None, None
        self.ra_at_xy_0, self.dec_at_xy_0 = None, None
        self.pixel_grid, self.transform_pix2angle, self.coords = None, None, None
