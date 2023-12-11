import math
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Plots import plot_util
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import data_util, util
from scipy.stats import norm, truncnorm, uniform


class Lens:
    def __init__(self, z_lens, z_source, theta_e, lens_x, lens_y, source_x, source_y, mag_lens, mag_source):
        # define redshifts and cosmology
        self.z_lens = z_lens
        self.z_source = z_source
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        self.lens_model_list = ['SIE', 'SHEAR']
        self.lens_redshift_list = [self.z_lens, self.z_lens]
        self.lens_model_class = LensModel(self.lens_model_list)
        kwargs_mass = {
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
        self.kwargs_lens = [kwargs_mass, kwargs_shear]

        # light model: sersic ellipse profile
        self.lens_light_model_list = ['SERSIC_ELLIPSE']
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
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
        self.source_model_class = LightModel(self.source_model_list)
        kwargs_sersic = {
            'magnitude': mag_source,
            'R_sersic': truncnorm(-2, 2, loc=0.35, scale=0.05).rvs(),
            'n_sersic': truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(),
            'e1': norm(loc=0.0, scale=0.1).rvs(),
            'e2': norm(loc=0.0, scale=0.1).rvs(),
            'center_x': source_x,
            'center_y': source_y
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
            'z_source_convention': 4,
            # source redshift to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
        }

        # set kwargs in terms of amp (converted from magnitude)
        self.kwargs_source_amp, self.kwargs_lens_light_amp = None, None
        self._set_amp_light_kwargs(self)

        self.delta_pix, self.num_pix = None, None
        self.ra_at_xy_0, self.dec_at_xy_0 = None, None
        self.Mpix2coord, self.Mcoord2pix = None, None
        self.pixel_grid, self.coords = None, None
        self.lenstronomy_roman_config = None


    # TODO something useful
    # def __str__(self):

    def add_subhalos(self, halo_lens_model_list, halo_redshift_list, kwargs_halos):
        self.lens_model_list += halo_lens_model_list
        self.lens_model_class = LensModel(self.lens_model_list)

        self.lens_redshift_list += halo_redshift_list
        self.kwargs_lens += kwargs_halos


    def update_classes(self):
        self.lens_model_class = LensModel(self.lens_model_list)
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
        self.source_model_class = LightModel(self.source_model_list)


    def update_model(self):
        # update model
        self.kwargs_model = {
            'lens_model_list': self.lens_model_list,
            'lens_redshift_list': self.lens_redshift_list,
            'lens_light_model_list': self.lens_light_model_list,
            'source_light_model_list': self.source_model_list,
            'source_redshift_list': self.source_redshift_list,
            'cosmo': self.cosmo,
            'z_source': self.z_source,
            'z_source_convention': 4,
            # source redshift to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
        }


    def get_array(self, num_pix, side, kwargs_psf={'psf_type': 'NONE'}):
        self.num_pix = num_pix
        self._set_up_pixel_grid(num_pix, side)

        # define PSF, e.g. kwargs_psf = {'psf_type': 'NONE'}, {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        psf_class = PSF(**kwargs_psf)

        # define numerics
        kwargs_numerics = {
            'supersampling_factor': 1,
            'supersampling_convolution': False
        }

        # convert from physical to lensing units
        # self._mass_physical_to_lensing_units()

        image_model = ImageModel(data_class=self.pixel_grid,
                                 psf_class=psf_class,
                                 lens_model_class=self.lens_model_class,
                                 source_model_class=self.source_model_class,
                                 lens_light_model_class=self.lens_light_model_class,
                                 kwargs_numerics=kwargs_numerics)

        return image_model.image(kwargs_lens=self.kwargs_lens,
                                 kwargs_source=self.kwargs_source_amp,
                                 kwargs_lens_light=self.kwargs_lens_light_amp)


    def get_source_pixel_coords(self):
        source_ra, source_dec = self.kwargs_lens[0]['center_x'], self.kwargs_lens[0]['center_y']
        return self.coords.map_coord2pix(ra=source_ra, dec=source_dec)


    def get_lens_pixel_coords(self):
        lens_ra, lens_dec = self.kwargs_source[0]['center_x'], self.kwargs_source[0]['center_y']
        return self.coords.map_coord2pix(ra=lens_ra, dec=lens_dec)
    

    def _mass_physical_to_lensing_units(self):
        sim_g = SimAPI(numpix=self.num_pix, kwargs_single_band=self.lenstronomy_roman_config, kwargs_model=self.kwargs_model)
        self.kwargs_lens_lensing_units = sim_g.physical2lensing_conversion(kwargs_mass=self.kwargs_lens)


    def _set_up_pixel_grid(self, num_pix, side):
        self.delta_pix = side / num_pix  # size of pixel in angular coordinates

        ra_grid, dec_grid, self.ra_at_xy_0, self.dec_at_xy_0, x_at_radec_0, y_at_radec_0, self.Mpix2coord, self.Mcoord2pix = util.make_grid_with_coordtransform(
            numPix=num_pix, 
            deltapix=self.delta_pix, 
            subgrid_res=1, 
            left_lower=False, 
            inverse=False)

        kwargs_pixel = {'nx': num_pix, 'ny': num_pix,  # number of pixels per axis
                        'ra_at_xy_0': self.ra_at_xy_0,
                        'dec_at_xy_0': self.dec_at_xy_0,
                        'transform_pix2angle': self.Mpix2coord}

        self.pixel_grid = PixelGrid(**kwargs_pixel)
        self.coords = Coordinates(self.Mpix2coord, self.ra_at_xy_0, self.dec_at_xy_0)


    def _set_amp_light_kwargs(self):
        self.lenstronomy_roman_config = Roman(band='F106',
                                                psf_type='PIXEL', 
                                                survey_mode='wide_area').kwargs_single_band()
        magnitude_zero_point = self.lenstronomy_roman_config.get('magnitude_zero_point')

        self.kwargs_lens_light_amp = data_util.magnitude2amplitude(self.lens_light_model_class, 
                                                                    self.kwargs_lens_light,
                                                                    magnitude_zero_point)
        self.kwargs_source_amp = data_util.magnitude2amplitude(self.source_model_class, 
                                                                self.kwargs_source,
                                                                magnitude_zero_point)
    