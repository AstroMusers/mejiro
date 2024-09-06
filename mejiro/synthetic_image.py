import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Util import data_util
from lenstronomy.Util import util as len_util


class SyntheticImage:
    def __init__(self, strong_lens, instrument, band, arcsec, oversample=1, pieces=False, debugging=True, **kwargs):
        # TODO assert band is valid for instrument
        # assert band in instrument.get_bands()

        self.strong_lens = strong_lens
        self.instrument = instrument
        self.band = band
        self.arcsec = arcsec
        self.oversample = oversample
        self.kwargs = kwargs
        self.debugging = debugging
        self.pieces = pieces

        # calculate surface brightness
        self._calculate_surface_brightness(pieces=pieces)

        if self.debugging: print(
            f'Initialized SyntheticImage for StrongLens {self.strong_lens.uid} by {self.instrument.name} in {self.band} band')

    def _calculate_surface_brightness(self, pieces=False, kwargs_psf=None):
        self._set_up_pixel_grid()

        # define PSF, e.g. kwargs_psf = {'psf_type': 'NONE'}, {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        if kwargs_psf is None:
            kwargs_psf = {'psf_type': 'NONE'}
        psf_class = PSF(**kwargs_psf)

        # define numerics
        kwargs_numerics = {
            'supersampling_factor': 1,  # TODO should figure out how to set this optimally
            'supersampling_convolution': True
        }

        # convert from physical to lensing units if necessary e.g. sigma_v specified instead of theta_E
        if 'theta_E' not in self.strong_lens.kwargs_lens[0]:
            self.strong_lens._mass_physical_to_lensing_units()

        # to create ImageModel, need LensModel and LightModel objects
        self.strong_lens._set_classes()

        image_model = ImageModel(data_class=self.pixel_grid,
                                 psf_class=psf_class,
                                 lens_model_class=self.strong_lens.lens_model_class,
                                 source_model_class=self.strong_lens.source_model_class,
                                 lens_light_model_class=self.strong_lens.lens_light_model_class,
                                 kwargs_numerics=kwargs_numerics)

        self._convert_magnitudes_to_lenstronomy_amps()
        kwargs_lens_light_amp = [self.strong_lens.kwargs_lens_light_amp_dict[self.band]]
        kwargs_source_amp = [self.strong_lens.kwargs_source_amp_dict[self.band]]

        self.image = image_model.image(kwargs_lens=self.strong_lens.kwargs_lens,
                                       kwargs_source=kwargs_source_amp,
                                       kwargs_lens_light=kwargs_lens_light_amp)

        if pieces:
            self.lens_surface_brightness = image_model.lens_surface_brightness(kwargs_lens_light_amp)
            self.source_surface_brightness = image_model.source_surface_brightness(kwargs_source_amp,
                                                                                   self.strong_lens.kwargs_lens)

    def _convert_magnitudes_to_lenstronomy_amps(self):
        # TODO this is also awful
        if self.instrument.name == 'Roman':
            self.magnitude_zero_point = self.instrument.get_zeropoint_magnitude(self.band, self.kwargs['sca'])
        elif self.instrument.name == 'HWO':
            self.magnitude_zero_point = self.instrument.get_zeropoint_magnitude(self.band)

        kwargs_lens_light_amp = data_util.magnitude2amplitude(self.strong_lens.lens_light_model_class,
                                                              [self.strong_lens.kwargs_lens_light_dict[self.band]],
                                                              self.magnitude_zero_point)

        kwargs_source_amp = data_util.magnitude2amplitude(self.strong_lens.source_model_class,
                                                          [self.strong_lens.kwargs_source_dict[self.band]],
                                                          self.magnitude_zero_point)

        self.strong_lens.kwargs_lens_light_amp_dict[self.band] = kwargs_lens_light_amp[0]
        self.strong_lens.kwargs_source_amp_dict[self.band] = kwargs_source_amp[0]

    def _set_up_pixel_grid(self):
        # TODO this is awful
        if self.instrument.name == 'Roman':
            self.native_pixel_scale = self.instrument.pixel_scale
        elif self.instrument.name == 'HWO':
            self.native_pixel_scale = self.instrument.get_pixel_scale(self.band)

        self.pixel_scale = self.native_pixel_scale / self.oversample
        self.num_pix = np.ceil(self.arcsec / self.pixel_scale).astype(int)
        self.arcsec = self.num_pix * self.pixel_scale
        if self.debugging: print(
            f'Computing on pixel grid of size {self.num_pix}x{self.num_pix} ({self.arcsec}\"x{self.arcsec}\") with pixel scale {self.pixel_scale} arcsec/pixel (natively {self.native_pixel_scale} arcsec/pixel oversampled by factor {self.oversample})')

        _, _, self.ra_at_xy_0, self.dec_at_xy_0, _, _, self.Mpix2coord, self.Mcoord2pix = (
            len_util.make_grid_with_coordtransform(
                numPix=self.num_pix,
                deltapix=self.pixel_scale,
                subgrid_res=1,
                left_lower=False,
                inverse=False))

        kwargs_pixel = {'nx': self.num_pix, 'ny': self.num_pix,  # number of pixels per axis
                        'ra_at_xy_0': self.ra_at_xy_0,
                        'dec_at_xy_0': self.dec_at_xy_0,
                        'transform_pix2angle': self.Mpix2coord}

        self.pixel_grid = PixelGrid(**kwargs_pixel)
        self.coords = Coordinates(self.Mpix2coord, self.ra_at_xy_0, self.dec_at_xy_0)

    # TODO resample method to change oversample factor
    # should just re-run _calculate_surface_brightness with new oversample factor instead of doing some kind of interpolation
    # def resample():
