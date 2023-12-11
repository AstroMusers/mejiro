from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from package.helpers.lens import Lens


class TestPhysicalLens(Lens):
    def __init__(self):
        # define redshifts and cosmology
        self.z_lens = 0.5
        self.z_source = 1.5
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        # lens_cosmo = LensCosmo(z_lens=self.z_lens, z_source=self.z_source, cosmo=cosmo)

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
            'z_source_convention': 3,
            # source redshift to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
        }

        # set kwargs in terms of amp (converted from magnitude)
        self.kwargs_source_amp, self.kwargs_lens_light_amp = None, None
        self._set_amp_light_kwargs()

        self.delta_pix, self.num_pix = None, None
        self.ra_at_xy_0, self.dec_at_xy_0 = None, None
        self.Mpix2coord, self.Mcoord2pix = None, None
        self.pixel_grid, self.coords = None, None
        self.lenstronomy_roman_config = None
