import numpy as np
import warnings
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Util import data_util
from lenstronomy.Util import util as lenstronomy_util


class SyntheticImage:

    DEFAULT_KWARGS_NUMERICS = {
        'supersampling_factor': 5,  # super sampling factor of (partial) high resolution ray-tracing
        'compute_mode': 'regular',  # 'regular' or 'adaptive'
        'supersampling_convolution': True,  # bool, if True, performs the supersampled convolution (either on regular or adaptive grid)
        'supersampling_kernel_size': None,  # size of the higher resolution kernel region (can be smaller than the original kernel). None leads to use the full size
        'flux_evaluate_indexes': None,  # bool mask, if None, it will evaluate all (sub) pixels
        'supersampled_indexes': None,  # bool mask of pixels to be computed in supersampled grid (only for adaptive mode)
        'compute_indexes': None,  # bool mask of pixels to be computed the PSF response (flux being added to). Only used for adaptive mode and can be set =likelihood mask.
        'point_source_supersampling_factor': 5,
    }
    DEFAULT_KWARGS_PSF = {
        'psf_type': 'NONE'
    }

    def __init__(self, 
                 strong_lens,
                 instrument,
                 band,
                 fov_arcsec=5,
                 instrument_params={},
                 kwargs_numerics={},
                 kwargs_psf={},
                 pieces=False,
                 verbose=True
                 ):
        
        # check band is valid for instrument
        if band not in instrument.bands:
            raise ValueError(f'Band "{band}" not valid for instrument {instrument.name}')
        
        # set up instrument params 
        if not instrument_params:
            instrument_params = instrument.default_params()
        else:
            instrument_params = instrument.validate_instrument_params(instrument_params)
        
        # set up attributes
        self.strong_lens = strong_lens
        self.instrument_name = instrument.name
        self.instrument_params = instrument_params
        self.band = band
        self.fov_arcsec = fov_arcsec
        self.pieces = pieces
        self.verbose = verbose

        # calculate size of scene
        self.pixel_scale = instrument.get_pixel_scale(self.band).value  # an Astropy Quantity with units arcsec / pix
        self.num_pix = np.ceil(self.fov_arcsec / self.pixel_scale).astype(int)
        if self.num_pix % 2 == 0:
            self.num_pix += 1  # make sure that final image will have odd number of pixels on a side
        self.fov_arcsec = self.num_pix * self.pixel_scale  # adjust fov (may differ from user-provided input)
        # TODO log these calculations and updates

        # set up pixel grid and coordinates
        x, y, self.ra_at_xy_0, self.dec_at_xy_0, x_at_radec_0, y_at_radec_0, self.Mpix2coord, self.Mcoord2pix = (
            lenstronomy_util.make_grid_with_coordtransform(
                numPix=self.num_pix,
                deltapix=self.pixel_scale,
                subgrid_res=1,
                left_lower=False,
                inverse=False))
        kwargs_pixel = {
            'nx': self.num_pix,
            'ny': self.num_pix,
            'ra_at_xy_0': self.ra_at_xy_0,
            'dec_at_xy_0': self.dec_at_xy_0,
            'transform_pix2angle': self.Mpix2coord
        }
        self.pixel_grid = PixelGrid(**kwargs_pixel)
        self.coords = Coordinates(self.Mpix2coord, self.ra_at_xy_0, self.dec_at_xy_0)

        # if AB magnitudes are provided, need to convert to lenstronomy amplitudes before ray-shooting
        if self.strong_lens.magnitudes:
            # retrieve zero-point magnitude
            if instrument.name == 'Roman':
                magnitude_zero_point = instrument.get_zeropoint_magnitude(self.band,
                                                                                    self.instrument_params['detector'])
            elif instrument.name == 'HWO':
                magnitude_zero_point = instrument.get_zeropoint_magnitude(self.band)

            # retrieve lens and source magnitudes
            lens_magnitude = self.strong_lens.magnitudes['lens'][band]
            source_magnitude = self.strong_lens.magnitudes['source'][band]

            # make sure the right magnitude is set in the kwargs
            self.strong_lens.kwargs_lens_light[0]['magnitude'] = lens_magnitude
            self.strong_lens.kwargs_source[0]['magnitude'] = source_magnitude

            # convert magnitudes to lenstronomy amplitudes
            self.strong_lens.kwargs_lens_light = data_util.magnitude2amplitude(light_model_class=self.strong_lens.lens_light_model,
                                                                  kwargs_light_mag=self.strong_lens.kwargs_lens_light,
                                                                  magnitude_zero_point=magnitude_zero_point)
            self.strong_lens.kwargs_source = data_util.magnitude2amplitude(light_model_class=self.strong_lens.source_light_model,
                                                                  kwargs_light_mag=self.strong_lens.kwargs_source,
                                                                  magnitude_zero_point=magnitude_zero_point)

        # set kwargs_numerics
        if not kwargs_numerics:
            kwargs_numerics = SyntheticImage.DEFAULT_KWARGS_NUMERICS
        elif 'compute_mode' not in kwargs_numerics:
            kwargs_numerics['compute_mode'] = 'regular'
        if kwargs_numerics['compute_mode'] == 'adaptive' and 'supersampled_indexes' not in kwargs_numerics.keys():
            if self.verbose: print('Building adaptive grid')
            self.supersampled_indexes = self.build_adaptive_grid(pad=40)
            kwargs_numerics['supersampled_indexes'] = self.supersampled_indexes
        if kwargs_numerics['supersampling_factor'] < 5:
            warnings.warn('Supersampling factor less than 5 may not be sufficient for accurate results, especially when convolving with a non-trivial PSF')
        self.kwargs_numerics = kwargs_numerics            

        # set kwargs_psf
        if not kwargs_psf:
            kwargs_psf = SyntheticImage.DEFAULT_KWARGS_PSF
        self.psf_class = PSF(**kwargs_psf)

        # ray-shoot
        image_model = ImageModel(data_class=self.pixel_grid,
                                 psf_class=self.psf_class,
                                 lens_model_class=self.strong_lens.lens_model,
                                 source_model_class=self.strong_lens.source_light_model,
                                 lens_light_model_class=self.strong_lens.lens_light_model,
                                 kwargs_numerics=kwargs_numerics)
        self.image = image_model.image(kwargs_lens=self.strong_lens.kwargs_lens,
                                       kwargs_source=self.strong_lens.kwargs_source,
                                       kwargs_lens_light=self.strong_lens.kwargs_lens_light,
                                       kwargs_ps=self.strong_lens.kwargs_ps,
                                       kwargs_extinction=self.strong_lens.kwargs_extinction,
                                       kwargs_special=self.strong_lens.kwargs_special,
                                       unconvolved=False, 
                                       source_add=True, 
                                       lens_light_add=True, 
                                       point_source_add=True)

        if self.pieces:
            self.lens_surface_brightness = image_model.lens_surface_brightness(kwargs_lens_light=self.strong_lens.kwargs_lens_light, unconvolved=True)
            self.source_surface_brightness = image_model.source_surface_brightness(kwargs_source=self.strong_lens.kwargs_source, kwargs_lens=self.strong_lens.kwargs_lens, kwargs_extinction=self.strong_lens.kwargs_extinction, kwargs_special=self.strong_lens.kwargs_special, unconvolved=True)
        else:
            self.lens_surface_brightness, self.source_surface_brightness = None, None

    def build_adaptive_grid(self, pad):
        image_positions = self.get_image_positions()
        if len(image_positions) == 0 or len(image_positions[0]) == 0 or len(image_positions[1]) == 0:
            raise ValueError(f"Failed to calculate image positions: {image_positions}")

        # calculate how far the images are from the center of the scene
        image_radii = []
        for x, y in zip(image_positions[0], image_positions[1]):
            image_radii.append(np.sqrt((x - (self.num_pix // 2)) ** 2 + (y - (self.num_pix // 2)) ** 2))

        x = np.linspace(-self.num_pix // 2, self.num_pix // 2, self.num_pix)
        y = np.linspace(-self.num_pix // 2, self.num_pix // 2, self.num_pix)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt((X - (self.strong_lens.kwargs_lens[0]['center_x'] / self.pixel_scale)) ** 2 + (
                Y - (self.strong_lens.kwargs_lens[0]['center_y'] / self.pixel_scale)) ** 2)

        min = np.min(image_radii) - pad
        if min < 0:
            min = 0
        max = np.max(image_radii) + pad
        if max > self.num_pix // 2:
            max = self.num_pix // 2

        return (distance >= min) & (distance <= max)

    def get_image_positions(self, pixel=True):
        image_x, image_y = self.strong_lens.get_image_positions()

        if pixel:
            if self.coords is None:
                self._set_up_pixel_grid()
            return self.coords.map_coord2pix(ra=image_x, dec=image_y)
        else:
            return image_x, image_y
        