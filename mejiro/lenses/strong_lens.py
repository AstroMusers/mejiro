from copy import deepcopy

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
from astropy.cosmology import default_cosmology
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import data_util
from lenstronomy.Util import util as len_util
from pyHalo.preset_models import CDM

from mejiro.utils import util


class StrongLens:
    def __init__(self, kwargs_model, kwargs_params, lens_mags, source_mags, lens_stellar_mass=None, lens_vel_disp=None,
                 snr=None, uid=None):
        # set z_source convention default
        self.z_source_convention = 5

        self.lens_stellar_mass = lens_stellar_mass
        self.lens_vel_disp = lens_vel_disp
        self.snr = snr
        self.uid = uid

        # calculate lens total mass and main halo mass
        if lens_stellar_mass is not None:
            # see Table 3, doi:10.1088/0004-637X/724/1/511
            # a = 0.81
            # b = 0.35
            # log_m_total_10 = (1 / a) * (np.log10(self.lens_stellar_mass / 1e10) - b)
            # self.lens_total_mass = np.power(10, log_m_total_10) * 1e10 * (100 / 32)
            # self.main_halo_mass = self.lens_total_mass - self.lens_stellar_mass

            # try calculating from dark matter fraction, because above method was giving negative main halo masses for some stellar masses
            # Table 6, assuming Chabirer IMF
            a = 0.13
            b = 0.54
            self.f_dm = a * np.log10(self.lens_stellar_mass / 1e11) + b
            self.main_halo_mass = self.lens_stellar_mass / (1 - self.f_dm)
        else:
            self.lens_total_mass = None
            self.main_halo_mass = None

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

        # this is only used as a record on the object, since the unpack method above actual sets the params that are used
        self.kwargs_params = kwargs_params

        # set lens_model_list, lens_light_model_list, source_light_model_list
        self._unpack_kwargs_model(kwargs_model)

        # set kwargs_model
        self.kwargs_model = self._set_model()

        # set lens_cosmo
        self._set_lens_cosmo()

        # set classes self.lens_light_model_class, self.source_model_class, self.lens_model_class
        self._set_classes()

        # calculate comoving distances (in Gpc)
        self.d_l = redshift_to_comoving_distance(self.z_lens, self.cosmo)
        self.d_s = redshift_to_comoving_distance(self.z_source, self.cosmo)
        self.d_ls = self.d_s - self.d_l

        # ASSUMING StrongLens will NOT be initialized with subhalos
        # then, we can grab the macrolens
        self.lens_model_list_macro = deepcopy(self.lens_model_list)
        self.kwargs_lens_macro = deepcopy(self.kwargs_lens)
        self.redshift_list_macro = deepcopy(self.lens_redshift_list)

        # additional fields to initialize
        self.kwargs_lens_light_amp_dict, self.kwargs_source_amp_dict = {}, {}
        self.oversample_factor = None
        self.side = None
        self.num_pix = None
        self.num_subhalos = None
        self.realization = None
        self.detector, self.detector_position = None, None
        self.galsim_rng = None
        self.ra, self.dec = None, None
        self.pyhalo_cosmo = None

    def get_lenstronomy_kwargs(self, band):
        # TODO finish
        kwargs_params = {}
        return self.kwargs_model, kwargs_params

    def get_total_kappa(self, num_pix, side):
        self._check_realization()  # ensure subhalos have been added

        # set cosmology by initializing pyHalo's Cosmology object, otherwise Colossus throws an error when calling realization.lensing_quantities()
        self._set_pyhalo_cosmology()

        halo_lens_model_list, halo_redshift_array, kwargs_halos, _ = self.realization.lensing_quantities(add_mass_sheet_correction=True)

        # halo_lens_model_list and kwargs_halos are lists, but halo_redshift_list is ndarray
        halo_redshift_list = list(halo_redshift_array)

        lens_model_total = LensModel(lens_model_list=self.lens_model_list_macro + halo_lens_model_list,
                                z_lens=self.z_lens,
                                z_source=self.z_source,
                                lens_redshift_list=self.redshift_list_macro + halo_redshift_list,
                                cosmo=self.cosmo,
                                multi_plane=False)
        kwargs_lens_total = self.kwargs_lens_macro + kwargs_halos
        
        return self._get_kappa_image(lens_model_total, kwargs_lens_total, num_pix, side)
        
    def get_subhalo_kappa(self, num_pix, side):
        # TODO docs: this method does NOT include the negative mass sheet correction, so returns the convergence map of just the subhalos

        self._check_realization()  # ensure subhalos have been added

        # set cosmology by initializing pyHalo's Cosmology object, otherwise Colossus throws an error when calling realization.lensing_quantities()
        self._set_pyhalo_cosmology()

        halo_lens_model_list, halo_redshift_array, kwargs_halos, _ = self.realization.lensing_quantities(add_mass_sheet_correction=False)

        # halo_lens_model_list and kwargs_halos are lists, but halo_redshift_list is ndarray
        halo_redshift_list = list(halo_redshift_array)

        lens_model_halos_only = LensModel(lens_model_list=halo_lens_model_list, 
                                          z_lens=self.z_lens, 
                                          z_source=self.z_source, 
                                          lens_redshift_list=halo_redshift_list, 
                                          cosmo=self.cosmo, 
                                          multi_plane=False)
        
        return self._get_kappa_image(lens_model_halos_only, kwargs_halos, num_pix, side)

    def get_macrolens_kappa(self, num_pix, side):
        lens_model_macro = LensModel(self.lens_model_list_macro)
        return self._get_kappa_image(lens_model_macro, self.kwargs_lens_macro, num_pix, side)


    def _get_kappa_image(self, lens_model, kwargs_lens, num_pix, side):
        _r = np.linspace(-side / 2, side / 2, num_pix)
        xx, yy = np.meshgrid(_r, _r)

        return lens_model.kappa(xx.ravel(), yy.ravel(), kwargs_lens).reshape(num_pix, num_pix)
    
    def _check_realization(self):
        if self.realization is None:
            raise ValueError('No subhalos have been added to this StrongLens object.')

    def get_einstein_radius(self):
        return self.kwargs_lens[0]['theta_E']

    def get_main_halo_mass(self):
        return einstein_radius_to_mass(self.get_einstein_radius(), self.z_lens, self.z_source, self.cosmo)

    def mass_in_einstein_radius(self):
        return self.lens_cosmo.mass_in_theta_E(self.get_einstein_radius())

    def generate_cdm_subhalos(self, log_mlow=6, log_mhigh=10, subhalo_cone=10, los_normalization=0, r_tidal=0.5,
                              sigma_sub=0.055):
        return CDM(self.z_lens,
                   self.z_source,
                   sigma_sub=sigma_sub,
                   log_mlow=log_mlow,
                   log_mhigh=log_mhigh,
                   log_m_host=np.log10(self.main_halo_mass),
                   r_tidal=r_tidal,
                   cone_opening_angle_arcsec=subhalo_cone,
                   LOS_normalization=los_normalization,
                   kwargs_cosmo=util.get_kwargs_cosmo(self.cosmo))
    
    def _set_pyhalo_cosmology(self):
        from pyHalo.Cosmology.cosmology import Cosmology
        Cosmology(astropy_instance=self.cosmo)

    def add_subhalos(self, realization):  # , return_stats=False, suppress_output=True
        # set cosmology by initializing pyHalo's Cosmology object, otherwise Colossus throws an error when calling realization.lensing_quantities()
        self._set_pyhalo_cosmology()

        # set some params on the StrongLens object
        self.realization = realization
        self.num_subhalos = len(realization.halos)

        # generate lenstronomy objects
        halo_lens_model_list, halo_redshift_array, kwargs_halos, _ = realization.lensing_quantities(add_mass_sheet_correction=True)

        # halo_lens_model_list and kwargs_halos are lists, but halo_redshift_array is ndarray
        halo_redshift_list = list(halo_redshift_array)

        # add subhalos to lenstronomy objects that model the strong lens
        self.kwargs_lens += kwargs_halos
        self.lens_redshift_list += halo_redshift_list
        self.lens_model_list += halo_lens_model_list
        self.lens_model_class = LensModel(self.lens_model_list)

        # TODO save the negative mass sheet correction kappa as an attribute on this object

        # # now, need to adjust the Einstein radius to account for the subhalos we're about to add
        # # convert Einstein radius to effective lensing mass
        # original_einstein_radius = self.get_einstein_radius()
        # effective_lensing_mass = einstein_radius_to_mass(original_einstein_radius, self.d_l, self.d_s, self.d_ls)

        # # calculate total mass of subhalos within Einstein radius and total mass of all subhalos
        # total_mass_subhalos_within_einstein_radius, total_subhalo_mass = 0, 0
        # for halo in realization.halos:
        #     total_subhalo_mass += halo.mass
        #     if np.sqrt(halo.x ** 2 + halo.y ** 2) < original_einstein_radius:
        #         total_mass_subhalos_within_einstein_radius += halo.mass

        # # subtract off the mass of the subhalos within the Einstein radius to get the adjusted lensing mass
        # adjusted_lensing_mass = effective_lensing_mass - total_mass_subhalos_within_einstein_radius

        # # convert back to Einstein radius
        # adjusted_einstein_radius = mass_to_einstein_radius(adjusted_lensing_mass, self.d_l, self.d_s, self.d_ls)

        # # update lens_kwargs with the adjusted Einstein radius
        # self.kwargs_lens[0]['theta_E'] = adjusted_einstein_radius

        # # calculate useful percentages
        # if not suppress_output or return_stats:
        #     # total subhalo mass can be zero and throw division by zero error
        #     try:
        #         percent_subhalo_mass_within_einstein_radius = (
        #                                                               total_mass_subhalos_within_einstein_radius / total_subhalo_mass) * 100
        #     except:
        #         percent_subhalo_mass_within_einstein_radius = 0
        #     percent_change_lensing_mass = util.percent_change(effective_lensing_mass, adjusted_lensing_mass)
        #     percent_change_einstein_radius = util.percent_change(original_einstein_radius, adjusted_einstein_radius)

        # if not suppress_output:
        #     print(f'Original Einstein radius: {original_einstein_radius:.4f} arcsec')
        #     print(f'Adjusted Einstein radius: {adjusted_einstein_radius:.4f} arcsec')
        #     print(f'Percent change of Einstein radius: {percent_change_einstein_radius:.2f}%')
        #     print('------------------------------------')
        #     print(f'Effective lensing mass: {effective_lensing_mass:.4e} M_Sun')
        #     print(f'Adjusted lensing mass: {adjusted_lensing_mass:.4e} M_Sun')
        #     print(f'Percent change of lensing mass: {percent_change_lensing_mass:.2f}%')
        #     print('------------------------------------')
        #     print(
        #         f'Total mass of CDM halos within Einstein radius: {total_mass_subhalos_within_einstein_radius:.4e} M_Sun')
        #     print(f'Total mass of CDM halos: {total_subhalo_mass:.4e} M_Sun')
        #     print(
        #         f'Percentage of total subhalo mass within Einstein radius: {percent_subhalo_mass_within_einstein_radius:.2f}%')
        #     print('\n')

        # if return_stats:
        #     return {
        #         'original_einstein_radius': original_einstein_radius,
        #         'adjusted_einstein_radius': adjusted_einstein_radius,
        #         'percent_change_einstein_radius': percent_change_einstein_radius,
        #         'effective_lensing_mass': effective_lensing_mass,
        #         'adjusted_lensing_mass': adjusted_lensing_mass,
        #         'percent_change_lensing_mass': percent_change_lensing_mass,
        #         'total_mass_subhalos_within_einstein_radius': total_mass_subhalos_within_einstein_radius,
        #         'total_subhalo_mass': total_subhalo_mass,
        #         'percent_subhalo_mass_within_einstein_radius': percent_subhalo_mass_within_einstein_radius
        #     }

    # TODO method to calculate snr and update snr attribute based on the full image
    # TODO put final image as an attribute on this class?

    def get_lens_flux_cps(self, band):
        if band not in self.kwargs_lens_light_amp_dict.keys():
            self._convert_magnitudes_to_lenstronomy_amps(band)
        
        lens_flux_cps = self.lens_light_model_class.total_flux([self.kwargs_lens_light_amp_dict[band]])
        lens_flux_cps_total = np.sum(lens_flux_cps)
        return lens_flux_cps_total

    def get_source_flux_cps(self, band):
        if band not in self.kwargs_source_amp_dict.keys():
            self._convert_magnitudes_to_lenstronomy_amps(band)
        
        source_flux_cps = self.source_model_class.total_flux([self.kwargs_source_amp_dict[band]])
        source_flux_cps_total = np.sum(source_flux_cps)
        return source_flux_cps_total

    def get_total_flux_cps(self, band):
        # TODO inefficient for this to call both methods if neither is initialized
        return self.get_lens_flux_cps(band) + self.get_source_flux_cps(band)

    def get_array(self, num_pix, side, band, kwargs_psf=None, return_pieces=False):
        self.num_pix = num_pix
        self.side = side
        self.oversample_factor = round((0.11 * self.num_pix) / self.side)
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

        self._convert_magnitudes_to_lenstronomy_amps(band)
        kwargs_lens_light_amp = [self.kwargs_lens_light_amp_dict[band]]
        kwargs_source_amp = [self.kwargs_source_amp_dict[band]]

        total_image = image_model.image(kwargs_lens=self.kwargs_lens,
                                        kwargs_source=kwargs_source_amp,
                                        kwargs_lens_light=kwargs_lens_light_amp)

        if return_pieces:
            lens_surface_brightness = image_model.lens_surface_brightness(kwargs_lens_light_amp)
            source_surface_brightness = image_model.source_surface_brightness(kwargs_source_amp, self.kwargs_lens)
            return total_image, lens_surface_brightness, source_surface_brightness
        else:
            return total_image

    @staticmethod
    def _get_amp_light_kwargs(magnitude_zero_point, light_model_class, kwargs_light):
        return data_util.magnitude2amplitude(light_model_class, [kwargs_light], magnitude_zero_point)

    @staticmethod
    def _build_kwargs_light_dict(mag_dict, kwargs_light):
        kwargs_light_dict = {}
        for band, mag in mag_dict.items():
            kwargs_light_band = deepcopy(kwargs_light[0])
            kwargs_light_band['magnitude'] = mag
            kwargs_light_dict[band] = kwargs_light_band
        return kwargs_light_dict

    @staticmethod
    def _validate_mags(lens_mags, source_mags):
        lens_bands = list(lens_mags.keys())
        source_bands = list(source_mags.keys())
        assert lens_bands == source_bands, 'Pair of lens and source magnitudes not available for all filters.'

    def _convert_magnitudes_to_lenstronomy_amps(self, band):
        if band in ['F087', 'F146']:
            survey_mode = 'microlensing'
        else:
            survey_mode = 'wide_area'
        lenstronomy_roman_config = Roman(band=band.upper(),
                                         psf_type='PIXEL',
                                         survey_mode=survey_mode).kwargs_single_band()
        magnitude_zero_point = lenstronomy_roman_config.get('magnitude_zero_point')

        kwargs_lens_light_amp = self._get_amp_light_kwargs(magnitude_zero_point, self.lens_light_model_class,
                                                           self.kwargs_lens_light_dict[band])
        kwargs_source_amp = self._get_amp_light_kwargs(magnitude_zero_point, self.source_model_class,
                                                       self.kwargs_source_dict[band])

        self.kwargs_lens_light_amp_dict[band] = kwargs_lens_light_amp[0]
        self.kwargs_source_amp_dict[band] = kwargs_source_amp[0]

    def _set_classes(self):
        self.lens_model_class = LensModel(self.lens_model_list)
        self.lens_light_model_class = LightModel(self.lens_light_model_list)
        self.source_model_class = LightModel(self.source_model_list)

    def _set_lens_cosmo(self):
        self.lens_cosmo = get_lens_cosmo(self.z_lens, self.z_source, self.cosmo)

    def _set_model(self):
        return {
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

    def get_source_pixel_coords(self, coords):
        # the kwargs_source_dict has key/value pairs band/kwargs_source
        # center_x and y are the same across all kwargs_source, so grab the first key/value pair
        first_key = next(iter(self.kwargs_source_dict))

        source_ra = self.kwargs_source_dict[first_key]['center_x']
        source_dec = self.kwargs_source_dict[first_key]['center_y']
        return coords.map_coord2pix(ra=source_ra, dec=source_dec)

    def get_lens_pixel_coords(self, coords):
        lens_ra, lens_dec = self.kwargs_lens[0]['center_x'], self.kwargs_lens[0]['center_y']
        return coords.map_coord2pix(ra=lens_ra, dec=lens_dec)

    def _mass_physical_to_lensing_units(self):
        sim_g = SimAPI(numpix=self.num_pix, kwargs_single_band=Roman().kwargs_single_band(),
                       kwargs_model=self.kwargs_model)
        self.kwargs_lens_lensing_units = sim_g.physical2lensing_conversion(kwargs_mass=self.kwargs_lens)

    def _set_up_pixel_grid(self):
        self.delta_pix = self.side / self.num_pix  # size of pixel in angular coordinates

        _, _, self.ra_at_xy_0, self.dec_at_xy_0, _, _, self.Mpix2coord, self.Mcoord2pix = (
            len_util.make_grid_with_coordtransform(
                numPix=self.num_pix,
                deltapix=self.delta_pix,
                subgrid_res=1,
                left_lower=False,
                inverse=False))

        kwargs_pixel = {'nx': self.num_pix, 'ny': self.num_pix,  # number of pixels per axis
                        'ra_at_xy_0': self.ra_at_xy_0,
                        'dec_at_xy_0': self.dec_at_xy_0,
                        'transform_pix2angle': self.Mpix2coord}

        self.pixel_grid = PixelGrid(**kwargs_pixel)
        self.coords = Coordinates(self.Mpix2coord, self.ra_at_xy_0, self.dec_at_xy_0)

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
            self.cosmo = default_cosmology.get()
        if 'z_source' in kwargs_model:
            self.z_source = kwargs_model['z_source']
        if 'z_source_convention' in kwargs_model:
            self.z_source_convention = kwargs_model['z_source_convention']

    def __str__(self):
        return f'StrongLens {self.uid}'

    # TODO overload __iter__()?
    # https://www.geeksforgeeks.org/python-__iter__-__next__-converting-object-iterator/
    def csv_row(self):
        return [self.uid, self.z_lens, self.z_source, self.lens_mass, self.lens_vel_disp]

    @staticmethod
    def get_csv_headers():
        # TODO this is a helper method for exporting to CSV using the __iter__() method
        return ['uid', 'z_lens', 'z_source', 'lens_mass', 'lens_vel']


def redshift_to_comoving_distance(redshift, cosmo):
    # TODO docs should include that unit is Gpc
    z = redshift * cu.redshift
    return z.to(u.Gpc, cu.redshift_distance(cosmo, kind='comoving')).value


# def einstein_radius_to_mass(einstein_radius, z_lens, z_source, cosmo):
#     velocity_dispersion = einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source, cosmo)
#     return velocity_dispersion_to_mass(velocity_dispersion)


# def mass_to_einstein_radius(mass, z_lens, z_source, cosmo):
#     velocity_dispersion = mass_to_velocity_dispersion(mass)
#     return velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source, cosmo)


def einstein_radius_to_velocity_dispersion(einstein_radius, z_lens, z_source, cosmo):
    # TODO docs: for a SIS profile
    lens_cosmo = get_lens_cosmo(z_lens, z_source, cosmo)
    return lens_cosmo.sis_theta_E2sigma_v(einstein_radius)


def velocity_dispersion_to_einstein_radius(velocity_dispersion, z_lens, z_source, cosmo):
    # TODO docs: for a SIS profile
    lens_cosmo = get_lens_cosmo(z_lens, z_source, cosmo)
    return lens_cosmo.sis_sigma_v2theta_E(velocity_dispersion)


def get_lens_cosmo(z_lens, z_source, cosmo):
    return LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)


# G = 4.3009e-12  # in Gpc * (km/s)^2 / M_sun
# C_SQUARED = 299792.458 ** 2  # in (km/s)^2
# C_SQUARED_ON_4G = C_SQUARED / (4 * G)  # in M_sun

def mass_to_einstein_radius(m, d_l, d_s, d_ls):
    return np.sqrt(m / np.power(10, 11.09)) * np.sqrt(d_ls / (d_l * d_s))  # 


def einstein_radius_to_mass(einstein_radius, d_l, d_s, d_ls):
    return ((d_l * d_s) / d_ls) * np.power(einstein_radius, 2) * np.power(10, 11.09)
