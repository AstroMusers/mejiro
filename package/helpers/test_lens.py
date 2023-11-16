import math
import numpy as np

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF


class TestLens:
    def __init__(self):
        # LENS
        # mass model: singular isothermal ellipsoid with a shear
        lens_model_list = ['SIE', 'SHEAR']
        self.lens_model_class = LensModel(lens_model_list)
        kwargs_spemd = {'theta_E': 1.7, 'e1': 0.07, 'e2': -0.03, 'center_x': 0.05, 'center_y': -0.25}
        kwargs_shear = {'gamma1': 0.1, 'gamma2': -0.05}
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]

        # light model: sersic ellipse profile
        lens_light_model_list = ['SERSIC_ELLIPSE']
        self.lens_light_model_class = LightModel(lens_light_model_list)
        kwargs_sersic_lens = {'amp': 10, 'R_sersic': 0.6, 'n_sersic': 2, 'e1': -0.1, 'e2': 0.1, 'center_x': 0.05,
                              'center_y': 0}
        self.kwargs_lens_light = [kwargs_sersic_lens]

        # SOURCE
        # light model: sersic ellipse profile
        source_model_list = ['SERSIC_ELLIPSE']
        self.source_model_class = LightModel(source_model_list)
        kwargs_sersic = {'amp': 4, 'R_sersic': 0.1, 'n_sersic': 1, 'e1': -0.1, 'e2': 0.1, 'center_x': 0.1,
                         'center_y': 0}
        self.kwargs_source = [kwargs_sersic]

        # set up model
        self.kwargs_model = {
            'lens_model_list': lens_model_list,
            'lens_light_model_list': lens_light_model_list,
            'source_light_model_list': source_model_list
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
        kwargs_numerics = {'supersampling_factor': 4,
                           'supersampling_convolution': False}

        image_model = ImageModel(data_class=pixel_grid,
                                 psf_class=psf_class,
                                 lens_model_class=self.lens_model_class,
                                 source_model_class=self.source_model_class,
                                 lens_light_model_class=self.lens_light_model_class,
                                 kwargs_numerics=kwargs_numerics)

        return image_model.image(kwargs_lens=self.kwargs_lens,
                                 kwargs_source=self.kwargs_source,
                                 kwargs_lens_light=self.kwargs_lens_light)
