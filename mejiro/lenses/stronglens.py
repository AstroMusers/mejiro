import astropy.cosmology as astropy_cosmo
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import data_util, util


class StrongLens:
    def __init__(self, kwargs_model, kwargs_params, band, uid=None):
        # set unique identifier
        self.uid = uid

        # set band
        self.band = band.lower()

        # get redshifts
        self.z_lens = kwargs_model['lens_redshift_list'][0]
        self.z_source = kwargs_model['z_source']

        # set kwargs_lens, kwargs_lens_light, kwargs_source
        self._unpack_kwargs_params(kwargs_params)

        # set lens_model_list, lens_light_model_list, source_light_model_list
        self._unpack_kwargs_model(kwargs_model)

        # set kwargs_model
        self.update_model()

    def _unpack_kwargs_params(self, kwargs_params):
        self.kwargs_lens = kwargs_params['kwargs_lens']
        self.kwargs_lens_light = kwargs_params['kwargs_lens_light']
        self.kwargs_source = kwargs_params['kwargs_source']

    def _unpack_kwargs_model(self, kwargs_model):
        # get model lists, which are required(TODO ?)
        self.lens_model_list = kwargs_model['lens_model_list']
        self.lens_light_model_list = kwargs_model['lens_light_model_list']
        self.source_model_list = kwargs_model['source_light_model_list']

        # set any optional key/value pairs
        self.lens_redshift_list, self.source_redshift_list, self.z_source = None, None, None

        if 'lens_redshift_list' in kwargs_model:
            self.lens_redshift_list = kwargs_model['lens_redshift_list']
        if 'source_redshift_list' in kwargs_model:
            self.source_redshift_list = kwargs_model['source_redshift_list']
        if 'cosmo' in kwargs_model:
            self.cosmo = kwargs_model['cosmo']
        else:
            self.cosmo = astropy_cosmo.default_cosmology.get()
        if 'z_source' in kwargs_model:
            self.z_source = kwargs_model['z_source']
        if 'z_source_convention' in kwargs_model:
            self.z_source_convention = kwargs_model['z_source_convention']
        else:
            self.z_source_convention = 4

    def add_subhalos(self, halo_lens_model_list, halo_redshift_list, kwargs_halos):
        # add subhalos to list of lensing objects
        self.kwargs_lens += kwargs_halos

        # set redshifts of subhalos
        self.lens_redshift_list += halo_redshift_list

        # update other lenstronomy objects
        self.lens_model_list += halo_lens_model_list
        self.lens_model_class = LensModel(self.lens_model_list)

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
            'supersampling_factor': 1,  # TODO crank this up?
            'supersampling_convolution': False
        }

        # convert from physical to lensing units if necessary e.g. sigma_v specified instead of theta_E
        if 'theta_E' not in self.kwargs_lens[0]:
            self._mass_physical_to_lensing_units()

        # to create ImageModel, need LensModel and LightModel objects
        self._set_classes()

        image_model = ImageModel(data_class=self.pixel_grid,
                                 psf_class=psf_class,
                                 lens_model_class=self.lens_model_class,
                                 source_model_class=self.source_model_class,
                                 lens_light_model_class=self.lens_light_model_class,
                                 kwargs_numerics=kwargs_numerics)

        # convert brightnesses to lenstronomy amp from magnitudes
        if 'magnitude' in self.kwargs_lens_light[0].keys():
            self._set_amp_light_kwargs()
        else:
            self.kwargs_lens_light_amp = self.kwargs_lens_light
            self.kwargs_source_amp = self.kwargs_source

        return image_model.image(kwargs_lens=self.kwargs_lens,
                                 kwargs_source=self.kwargs_source_amp,
                                 kwargs_lens_light=self.kwargs_lens_light_amp)

    def _set_classes(self):
        self.lens_model_class = LensModel(self.lens_model_list)
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
        self.source_model_class = LightModel(self.source_model_list)

    def get_source_pixel_coords(self):
        source_ra, source_dec = self.kwargs_source[0]['center_x'], self.kwargs_source[0]['center_y']
        return self.coords.map_coord2pix(ra=source_ra, dec=source_dec)

    def get_lens_pixel_coords(self):
        lens_ra, lens_dec = self.kwargs_lens[0]['center_x'], self.kwargs_lens[0]['center_y']
        return self.coords.map_coord2pix(ra=lens_ra, dec=lens_dec)

    def _mass_physical_to_lensing_units(self):
        sim_g = SimAPI(numpix=self.num_pix, kwargs_single_band=self.lenstronomy_roman_config,
                       kwargs_model=self.kwargs_model)
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
        self.lenstronomy_roman_config = Roman(band=self.band.upper(),
                                              psf_type='PIXEL',
                                              survey_mode='wide_area').kwargs_single_band()
        magnitude_zero_point = self.lenstronomy_roman_config.get('magnitude_zero_point')

        self.kwargs_lens_light_amp = data_util.magnitude2amplitude(self.lens_light_model_class,
                                                                   self.kwargs_lens_light,
                                                                   magnitude_zero_point)
        self.kwargs_source_amp = data_util.magnitude2amplitude(self.source_model_class,
                                                               self.kwargs_source,
                                                               magnitude_zero_point)

    # TODO something useful
    # def __str__(self):
