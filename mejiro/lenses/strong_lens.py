import astropy.cosmology as astropy_cosmo
from copy import deepcopy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
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
    def __init__(self, kwargs_model, kwargs_params, lens_mags, source_mags, uid=None):
        # set z_source convention default
        self.z_source_convention = 5

        # set unique identifier
        self.uid = uid

        # get redshifts
        self.z_lens = kwargs_model['lens_redshift_list'][0]
        self.z_source = kwargs_model['z_source']

        # set magnitudes in each band
        self.lens_mags = lens_mags
        self.source_mags = source_mags

        # confirm that magnitudes are set up correctly
        self._validate_mags(self.lens_mags, self.source_mags)

        # set light kwargs dicts and set kwargs_lens
        self._unpack_kwargs_params(kwargs_params)
        
        # set lens_model_list, lens_light_model_list, source_light_model_list
        self._unpack_kwargs_model(kwargs_model)

        # set kwargs_model TODO is this necessary?
        self.update_model()

        # set place to store amp versions of kwargs_light dicts
        self.kwargs_lens_light_amp_dict, self.kwargs_source_amp_dict = {}, {}

    def _build_kwargs_light_dict(self, mag_dict, kwargs_light):
        kwargs_light_dict = {}
        for band, mag in mag_dict.items():
            kwargs_light_band = deepcopy(kwargs_light[0])
            kwargs_light_band['magnitude'] = mag
            kwargs_light_dict[band] = kwargs_light_band
        return kwargs_light_dict

    def _validate_mags(self, lens_mags, source_mags):
        lens_bands = list(lens_mags.keys())
        source_bands = list(source_mags.keys())
        assert lens_bands == source_bands, 'Pair of lens and source magnitudes not available for all filters.'

    def _unpack_kwargs_params(self, kwargs_params):
        # set kwargs_lens
        self.kwargs_lens = kwargs_params['kwargs_lens']

        # set light dicts
        kwargs_lens_light = kwargs_params['kwargs_lens_light']
        kwargs_source = kwargs_params['kwargs_source']
        self.kwargs_lens_light_dict = self._build_kwargs_light_dict(self.lens_mags, kwargs_lens_light)
        self.kwargs_source_dict = self._build_kwargs_light_dict(self.source_mags, kwargs_source)

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


    def get_main_halo_mass(self):
        # instantiate LensCosmo object which handles unit conversions given a lens plane
        lens_cosmo = LensCosmo(z_lens=self.z_lens, z_source=self.z_source, cosmo=self.cosmo)

        # return the main halo mass in units of solar masses
        return lens_cosmo.mass_in_theta_E(theta_E=self.kwargs_lens[0]['theta_E'])


    def add_subhalos(self, halo_lens_model_list, halo_redshift_list, kwargs_halos, total_subhalo_mass):
        # add subhalos to list of lensing objects
        self.kwargs_lens += kwargs_halos

        # set redshifts of subhalos
        self.lens_redshift_list += halo_redshift_list

        # update other lenstronomy objects
        self.lens_model_list += halo_lens_model_list
        self.lens_model_class = LensModel(self.lens_model_list)

        # TODO update the mass of the main halo to account for the added subhalos
        main_halo_mass = self.get_main_halo_mass()

        # TODO calculate updated Einstein radius based on total subhalo mass
        # updated_theta_E = 

        # update lens_kwargs with the adjusted Einstein radius
        # self.kwargs_lens[0]['theta_E'] = updated_theta_E


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
            'z_source_convention': self.z_source_convention,
            # source redshift to which the reduced deflections are computed, is the maximal redshift of the ray-tracing
        }

    def get_lens_flux_cps(self, band):
        return self.lens_light_model_class.total_flux([self.kwargs_lens_light_amp_dict[band]])[0]

    def get_source_flux_cps(self, band):
        return self.source_model_class.total_flux([self.kwargs_source_amp_dict[band]])[0]
    
    def get_total_flux_cps(self, band):
        return self.get_lens_flux_cps(band) + self.get_source_flux_cps(band)

    def get_array(self, num_pix, side, band, kwargs_psf={'psf_type': 'NONE'}):
        self.num_pix = num_pix
        self.side = side
        self.oversample_factor = round((0.11 * self.num_pix) / self.side)
        self._set_up_pixel_grid()

        # define PSF, e.g. kwargs_psf = {'psf_type': 'NONE'}, {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        psf_class = PSF(**kwargs_psf)

        # define numerics
        kwargs_numerics = {
            'supersampling_factor': 1,
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
        self.lenstronomy_roman_config = Roman(band=band.upper(),
                                              psf_type='PIXEL',
                                              survey_mode='wide_area').kwargs_single_band()
        magnitude_zero_point = self.lenstronomy_roman_config.get('magnitude_zero_point')

        kwargs_lens_light_amp = self._get_amp_light_kwargs(magnitude_zero_point, self.lens_light_model_class, self.kwargs_lens_light_dict[band])
        kwargs_source_amp = self._get_amp_light_kwargs(magnitude_zero_point, self.source_model_class, self.kwargs_source_dict[band])

        self.kwargs_lens_light_amp_dict[band] = kwargs_lens_light_amp[0]
        self.kwargs_source_amp_dict[band] = kwargs_source_amp[0]

        return image_model.image(kwargs_lens=self.kwargs_lens,
                                 kwargs_source=kwargs_source_amp,
                                 kwargs_lens_light=kwargs_lens_light_amp)
    
    def _get_amp_light_kwargs(self, magnitude_zero_point, light_model_class, kwargs_light):
        return data_util.magnitude2amplitude(light_model_class, [kwargs_light], magnitude_zero_point)

    def _set_classes(self):
        self.lens_model_class = LensModel(self.lens_model_list)
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
        self.source_model_class = LightModel(self.source_model_list)

    def get_source_pixel_coords(self):  # TODO need to test this
        first_key = next(iter(self.kwargs_source_dict))
        source_ra = self.kwargs_source_dict[first_key][0]['center_x']
        source_dec = self.kwargs_source_dict[first_key][0]['center_y']
        return self.coords.map_coord2pix(ra=source_ra, dec=source_dec)

    def get_lens_pixel_coords(self):
        lens_ra, lens_dec = self.kwargs_lens[0]['center_x'], self.kwargs_lens[0]['center_y']
        return self.coords.map_coord2pix(ra=lens_ra, dec=lens_dec)

    def _mass_physical_to_lensing_units(self):
        sim_g = SimAPI(numpix=self.num_pix, kwargs_single_band=self.lenstronomy_roman_config,
                       kwargs_model=self.kwargs_model)
        self.kwargs_lens_lensing_units = sim_g.physical2lensing_conversion(kwargs_mass=self.kwargs_lens)

    def _set_up_pixel_grid(self):
        self.delta_pix = self.side / self.num_pix  # size of pixel in angular coordinates

        ra_grid, dec_grid, self.ra_at_xy_0, self.dec_at_xy_0, x_at_radec_0, y_at_radec_0, self.Mpix2coord, self.Mcoord2pix = util.make_grid_with_coordtransform(
            numPix=self.num_pix,
            deltapix=self.delta_pix,
            subgrid_res=1,
            left_lower=False,
            inverse=False)

        kwargs_pixel = {'nx': self.num_pix, 'ny': self.num_pix,  # number of pixels per axis
                        'ra_at_xy_0': self.ra_at_xy_0,
                        'dec_at_xy_0': self.dec_at_xy_0,
                        'transform_pix2angle': self.Mpix2coord}

        self.pixel_grid = PixelGrid(**kwargs_pixel)
        self.coords = Coordinates(self.Mpix2coord, self.ra_at_xy_0, self.dec_at_xy_0)

    def __str__(self):
        return f'StrongLens {self.uid}'