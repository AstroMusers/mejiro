import math
import numpy as np
from copy import deepcopy

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.Plots import plot_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import data_util


class TestPhysicalLens:
    def __init__(self):
        # define redshifts and cosmology
        self.z_lens = 0.5
        self.z_source = 1.5
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        # lens_cosmo = LensCosmo(z_lens=self.z_lens, z_source=self.z_source, cosmo=cosmo)

        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        lens_model_list = ['SIE', 'SHEAR']
        lens_redshift_list = [self.z_lens, self.z_lens]
        self.lens_model_class = LensModel(lens_model_list)
        kwargs_mass = {
            'sigma_v': 250,  # velocity dispersion in units km/s
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
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.lens_light_model_class = LightModel(lens_light_model_list)
        kwargs_sersic_lens = {
            'magnitude': 23, 
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
        source_model_list = ['SERSIC_ELLIPSE']
        source_redshift_list = [self.z_source]
        self.source_model_class = LightModel(source_model_list)
        kwargs_sersic = {
            'magnitude': 27, 
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
            'lens_model_list': lens_model_list,
            'lens_redshift_list': lens_redshift_list,
            'lens_light_model_list': lens_light_model_list,
            'source_light_model_list': source_model_list,
            'source_redshift_list': source_redshift_list,
            'cosmo': cosmo,
            'z_source': self.z_source,
            'z_source_convention': 3,  # source redshift to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
        }

        self.ra_at_xy_0, self.dec_at_xy_0 = None, None

    def get_array(self, num_pix, side=5.):
        delta_pix = side / num_pix  # size of pixel in angular coordinates

        # specify coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
        self.ra_at_xy_0, self.dec_at_xy_0 = -delta_pix * math.ceil(num_pix / 2), -delta_pix * math.ceil(num_pix / 2)

        # define linear translation matrix of a shift in pixel in a shift in coordinates
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix

        kwargs_pixel = {'nx': num_pix, 'ny': num_pix,  # number of pixels per axis
                        'ra_at_xy_0': self.ra_at_xy_0,
                        'dec_at_xy_0': self.dec_at_xy_0,
                        'transform_pix2angle': transform_pix2angle}
        pixel_grid = PixelGrid(**kwargs_pixel)

        # define PSF
        kwargs_psf = {'psf_type': 'NONE'}
        psf_class = PSF(**kwargs_psf)

        # define numerics
        kwargs_numerics = {
            'supersampling_factor': 1,
            'supersampling_convolution': False
        }

        # convert from magnitude to amplitude for light models
        lenstronomy_roman_config = Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area').kwargs_single_band()
        magnitude_zero_point = lenstronomy_roman_config.get('magnitude_zero_point')
        kwargs_lens_light_amp = data_util.magnitude2amplitude(self.lens_light_model_class, self.kwargs_lens_light, magnitude_zero_point)
        kwargs_source_amp = data_util.magnitude2amplitude(self.source_model_class, self.kwargs_source, magnitude_zero_point)

        # convert from physical to lensing units
        sim_g = SimAPI(numpix=num_pix, kwargs_single_band=lenstronomy_roman_config, kwargs_model=self.kwargs_model)
        kwargs_lens_lensing_units = sim_g.physical2lensing_conversion(kwargs_mass=self.kwargs_lens)

        image_model = ImageModel(data_class=pixel_grid,
                                 psf_class=psf_class,
                                 lens_model_class=self.lens_model_class,
                                 source_model_class=self.source_model_class,
                                 lens_light_model_class=self.lens_light_model_class,
                                 kwargs_numerics=kwargs_numerics)

        return image_model.image(kwargs_lens=kwargs_lens_lensing_units,
                                 kwargs_source=kwargs_source_amp,
                                 kwargs_lens_light=kwargs_lens_light_amp)
    

    def get_roman_sim(self, side=5.):
        kwargs_numerics = {'point_source_supersampling_factor': 1}

        roman_g = Roman(band='F062', psf_type='PIXEL', survey_mode='wide_area')
        roman_r = Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')
        roman_i = Roman(band='F184', psf_type='PIXEL', survey_mode='wide_area')

        kwargs_b_band = roman_g.kwargs_single_band()
        kwargs_g_band = roman_r.kwargs_single_band()
        kwargs_r_band = roman_i.kwargs_single_band()

        # set number of pixels from pixel scale
        pixel_scale = kwargs_g_band['pixel_scale']
        numpix = int(round(side / pixel_scale))

        sim_b = SimAPI(numpix=numpix, kwargs_single_band=kwargs_b_band, kwargs_model=self.kwargs_model)
        sim_g = SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=self.kwargs_model)
        sim_r = SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=self.kwargs_model)

        imsim_b = sim_b.image_model_class(kwargs_numerics)
        imsim_g = sim_g.image_model_class(kwargs_numerics)
        imsim_r = sim_r.image_model_class(kwargs_numerics)

        # TODO TEMP for now, just set flat spectrum i.e. same brightness across filters
        kwargs_lens_light_mag_g = self.kwargs_lens_light
        kwargs_lens_light_mag_r = self.kwargs_lens_light
        kwargs_lens_light_mag_i = self.kwargs_lens_light
        kwargs_source_mag_g = self.kwargs_source
        kwargs_source_mag_r = self.kwargs_source
        kwargs_source_mag_i = self.kwargs_source

        # turn magnitude kwargs into lenstronomy kwargs
        kwargs_lens_light_g, kwargs_source_g, _ = sim_b.magnitude2amplitude(kwargs_lens_light_mag_g, kwargs_source_mag_g, None)
        kwargs_lens_light_r, kwargs_source_r, _ = sim_g.magnitude2amplitude(kwargs_lens_light_mag_r, kwargs_source_mag_r, None)
        kwargs_lens_light_i, kwargs_source_i, _ = sim_r.magnitude2amplitude(kwargs_lens_light_mag_i, kwargs_source_mag_i, None)

        # convert from physical to lensing units
        kwargs_lens_lensing_units = sim_b.physical2lensing_conversion(kwargs_mass=self.kwargs_lens)

        image_b = imsim_b.image(kwargs_lens_lensing_units, kwargs_source_g, kwargs_lens_light_g)
        image_g = imsim_g.image(kwargs_lens_lensing_units, kwargs_source_r, kwargs_lens_light_r)
        image_r = imsim_r.image(kwargs_lens_lensing_units, kwargs_source_i, kwargs_lens_light_i)

        # add noise
        image_b += sim_b.noise_for_model(model=image_b)
        image_g += sim_g.noise_for_model(model=image_g)
        image_r += sim_r.noise_for_model(model=image_r)

        image = image_g
        rgb_image = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)

        # scale_max=10000
        def _scale_max(image):
            flat = image.flatten()
            flat.sort()
            scale_max = flat[int(len(flat) * 0.95)]
            return scale_max

        rgb_image[:, :, 0] = plot_util.sqrt(image_b, scale_min=0, scale_max=_scale_max(image_b))
        rgb_image[:, :, 1] = plot_util.sqrt(image_g, scale_min=0, scale_max=_scale_max(image_g))
        rgb_image[:, :, 2] = plot_util.sqrt(image_r, scale_min=0, scale_max=_scale_max(image_r))

        return image, rgb_image, sim_b.data_class
