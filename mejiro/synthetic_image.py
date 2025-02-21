import numpy as np
import warnings
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Util import data_util
from lenstronomy.Util import util as len_util


class SyntheticImage:
    def __init__(self, strong_lens, instrument, band, arcsec, oversample=5,
                 kwargs_numerics={}, pieces=False, verbose=True,
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

        # if kwargs_numerics is empty, use default values; defaulting dictionary in init gives weirdness
        KWARGS_NUMERICS_DEFAULT = {'supersampling_factor': 3, 'compute_mode': 'adaptive'}
        if not kwargs_numerics:
            kwargs_numerics = KWARGS_NUMERICS_DEFAULT
        else:
            # check for extra keys
            for key in kwargs_numerics.keys():
                if key not in ['supersampling_factor', 'compute_mode', 'supersampled_indexes']:
                    warnings.warn(f'Unrecognized key in kwargs_numerics: {key}')
            
            # do some defaulting if certain keys are left out
            if 'supersampling_factor' not in kwargs_numerics.keys():
                kwargs_numerics['supersampling_factor'] = KWARGS_NUMERICS_DEFAULT['supersampling_factor']
            if 'compute_mode' not in kwargs_numerics.keys():
                kwargs_numerics['compute_mode'] = KWARGS_NUMERICS_DEFAULT['compute_mode']

        # build adaptive grid
        if kwargs_numerics['compute_mode'] == 'adaptive' and 'supersampled_indexes' not in kwargs_numerics.keys():
            if self.verbose: print('Building adaptive grid')
            self.supersampled_indexes = self.build_adaptive_grid(pad=40)
            kwargs_numerics['supersampled_indexes'] = self.supersampled_indexes

        if self.verbose:
            print(
                f'Computing with \'{kwargs_numerics["compute_mode"]}\' mode and supersampling factor {kwargs_numerics["supersampling_factor"]}')
            if kwargs_numerics['compute_mode'] == 'adaptive':
                print(f'Adaptive grid: {self.supersampled_indexes.shape}')

        self._calculate_surface_brightness(kwargs_numerics, pieces)

        if self.verbose: print(
            f'Initialized SyntheticImage for StrongLens {self.strong_lens.uid} by {self.instrument.name} in {self.band} band')

    def build_adaptive_grid(self, pad):
        image_positions = self.get_image_positions()
        if len(image_positions) == 0 or len(image_positions[0]) == 0 or len(image_positions[1]) == 0:
            raise ValueError(f"Image positions are empty: {image_positions}")

        image_radii = []
        for x, y in zip(image_positions[0], image_positions[1]):
            image_radii.append(np.sqrt((x - (self.num_pix // 2)) ** 2 + (y - (self.num_pix // 2)) ** 2))

        if len(image_radii) == 0:
            raise ValueError(f"Image radii list is empty: {image_radii}")

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

    def get_image_positions(self):
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        try:
            first_key = next(iter(self.strong_lens.kwargs_source_dict))  # get first key from source dict
        except StopIteration:
            raise ValueError("kwargs_source_dict is empty.")

        source_x = self.strong_lens.kwargs_source_dict[first_key]['center_x']
        source_y = self.strong_lens.kwargs_source_dict[first_key]['center_y']

        solver = LensEquationSolver(self.strong_lens.lens_model_class)
        image_x, image_y = solver.image_position_from_source(sourcePos_x=source_x, sourcePos_y=source_y,
                                                             kwargs_lens=self.strong_lens.kwargs_lens)

        if self.coords is None:
            self._set_up_pixel_grid()

        return self.coords.map_coord2pix(ra=image_x, dec=image_y)

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

    def _calculate_surface_brightness(self, kwargs_numerics, pieces=False, kwargs_psf=None):
        # define PSF, e.g. kwargs_psf = {'psf_type': 'NONE'}, {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        if kwargs_psf is None:
            kwargs_psf = {'psf_type': 'NONE'}
        psf_class = PSF(**kwargs_psf)

        # convert from physical to lensing units if necessary e.g. sigma_v specified instead of theta_E
        if 'theta_E' not in self.strong_lens.kwargs_lens[0]:
            self.strong_lens._mass_physical_to_lensing_units()

        # to create ImageModel, need LensModel and LightModel objects
        self.strong_lens._set_classes()

        image_model = ImageModel(data_class=self.pixel_grid,
                                 psf_class=psf_class,
                                 # TODO does this need to be passed in? I never want to convolve with a PSF at this stage
                                 lens_model_class=self.strong_lens.lens_model_class,
                                 source_model_class=self.strong_lens.source_model_class,
                                 lens_light_model_class=self.strong_lens.lens_light_model_class,
                                 kwargs_numerics=kwargs_numerics)

        # if kwargs_lens_light_amp_dict and kwargs_source_amp_dict are empty, it means magnitudes were provided, so convert magnitudes to lenstronomy amps
        if not self.strong_lens.kwargs_lens_light_amp_dict and not self.strong_lens.kwargs_source_amp_dict:
            self._convert_magnitudes_to_lenstronomy_amps()

        # lenstronomy kwargs are lists
        kwargs_lens_light_amp = self.strong_lens.kwargs_lens_light_amp_dict[self.band]
        kwargs_source_amp = self.strong_lens.kwargs_source_amp_dict[self.band]

        # ray-shoot
        self.image = image_model.image(kwargs_lens=self.strong_lens.kwargs_lens,
                                       kwargs_source=kwargs_source_amp,
                                       kwargs_lens_light=kwargs_lens_light_amp,
                                       unconvolved=True)

        if pieces:
            self.lens_surface_brightness = image_model.lens_surface_brightness(kwargs_lens_light_amp, unconvolved=True)
            self.source_surface_brightness = image_model.source_surface_brightness(kwargs_source_amp,
                                                                                   self.strong_lens.kwargs_lens,
                                                                                   unconvolved=True)
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
