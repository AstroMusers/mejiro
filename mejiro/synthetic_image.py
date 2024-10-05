import warnings

import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Util import data_util
from lenstronomy.Util import util as len_util


class SyntheticImage:
    def __init__(self, strong_lens, instrument, band, arcsec, oversample=1, pieces=False, verbose=True,
                 instrument_params={}):
        # assert band is valid for instrument
        assert band in instrument.bands, f'Band "{band}" not valid for instrument {instrument.name}'

        # default instrument params if none, else validate
        if not instrument_params:
            self.instrument_params = instrument.default_params()
        else:
            self.instrument_params = instrument.validate_instrument_params(instrument_params)

        self.strong_lens = strong_lens
        self.instrument = instrument
        self.band = band
        self.verbose = verbose
        self.pieces = pieces

        self._set_up_pixel_grid(arcsec, oversample)
        self._calculate_surface_brightness(pieces=pieces)

        if self.verbose: print(
            f'Initialized SyntheticImage for StrongLens {self.strong_lens.uid} by {self.instrument.name} in {self.band} band')

    def _set_up_pixel_grid(self, arcsec, oversample):
        # validation for oversample
        self.oversample = int(oversample)
        assert self.oversample >= 1, 'Oversampling factor must be greater than 1'
        assert self.oversample % 2 == 1, 'Oversampling factor must be an odd integer'
        if oversample < 5:
            warnings.warn(
                'Oversampling factor less than 5 may not be sufficient for accurate results, especially when convolving with a non-trivial PSF')

        # "native" refers to the final output, as opposed to the oversampled grid that the intermediary calculations are performed on for better accuracy
        self.native_pixel_scale = self.instrument.get_pixel_scale(self.band)
        self.native_num_pix = np.ceil(arcsec / self.native_pixel_scale).astype(int)

        # make sure that final image will have odd number of pixels on a side
        if self.native_num_pix % 2 == 0:
            self.native_num_pix += 1

        # these parameters are for the oversampled grid
        self.pixel_scale = self.native_pixel_scale / self.oversample
        self.num_pix = self.native_num_pix * self.oversample

        # finally, adjust arcseconds (may differ from user-provided input)
        self.arcsec = self.native_num_pix * self.native_pixel_scale

        if self.verbose: print(
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

    def _calculate_surface_brightness(self, pieces=False, kwargs_psf=None):
        # define PSF, e.g. kwargs_psf = {'psf_type': 'NONE'}, {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        if kwargs_psf is None:
            kwargs_psf = {'psf_type': 'NONE'}
        psf_class = PSF(**kwargs_psf)

        # define numerics
        kwargs_numerics = {
            'supersampling_factor': 1,
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
        else:
            self.lens_surface_brightness, self.source_surface_brightness = None, None

    def _convert_magnitudes_to_lenstronomy_amps(self):
        if self.instrument.name == 'Roman':
            self.magnitude_zero_point = self.instrument.get_zeropoint_magnitude(self.band,
                                                                                self.instrument_params['detector'])
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

    def set_native_coords(self):
        _, _, self.ra_at_xy_0_native, self.dec_at_xy_0_native, _, _, self.Mpix2coord_native, self.Mcoord2pix_native = (
            len_util.make_grid_with_coordtransform(
                numPix=np.ceil(self.arcsec / self.native_pixel_scale).astype(int),
                deltapix=self.native_pixel_scale,
                subgrid_res=1,
                left_lower=False,
                inverse=False))
        self.coords_native = Coordinates(self.Mpix2coord_native, self.ra_at_xy_0_native, self.dec_at_xy_0_native)

    # TODO resample method to change oversample factor
    # should just re-run _calculate_surface_brightness with new oversample factor instead of doing some kind of interpolation
    # def resample():
