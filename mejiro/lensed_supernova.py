import numpy as np
from scipy.interpolate import interp1d

from mejiro.galaxy_galaxy import GalaxyGalaxy


class LensedSupernova(GalaxyGalaxy):

    def __init__(
            self,
            name,
            coords,
            kwargs_model,
            kwargs_params,
            physical_params={},
            use_jax=None
    ):
        super().__init__(name=name,
                         coords=coords,
                         kwargs_model=kwargs_model,
                         kwargs_params=kwargs_params,
                         physical_params=physical_params,
                         use_jax=use_jax)

        self.sn_type = physical_params.get('sn_type')
        self.light_curves = physical_params.get('light_curves', {})

    def get_time_delays(self):
        if 'time_delays' not in self.physical_params:
            raise ValueError("Time delays not found in physical_params.")
        return self.physical_params['time_delays']

    def get_point_source_magnification(self):
        if 'image_magnifications' not in self.physical_params:
            raise ValueError("Image magnifications not found in physical_params.")
        return self.physical_params['image_magnifications']

    def get_sn_image_positions(self):
        if not self.kwargs_ps:
            raise ValueError("No point source parameters found.")
        ra_image = self.kwargs_ps[0]['ra_image']
        dec_image = self.kwargs_ps[0]['dec_image']
        return ra_image, dec_image

    def get_light_curve(self, band):
        if band not in self.light_curves:
            raise ValueError(f"No light curve found for band '{band}'. "
                             f"Available bands: {list(self.light_curves.keys())}")
        return self.light_curves[band]

    def set_observation_time(self, time, band):
        """Interpolate stored light curves to set point source magnitudes at a
        specific observation time. Must be called before creating a SyntheticImage.

        Parameters
        ----------
        time : float
            Observation time in days.
        band : str
            Imaging band.
        """
        lc = self.get_light_curve(band)
        time_array = lc['time']
        magnitudes_per_image = lc['magnitudes']

        interpolated_mags = []
        for image_mags in magnitudes_per_image:
            interp_func = interp1d(time_array, image_mags,
                                   kind='linear', fill_value='extrapolate')
            interpolated_mags.append(float(interp_func(time)))

        self.kwargs_ps[0]['magnitude'] = interpolated_mags

    @staticmethod
    def from_slsim(slsim_lens, name=None, coords=None, bands=None, use_jax=None):
        cosmo = slsim_lens.cosmo
        z_lens = slsim_lens.deflector_redshift
        z_source = slsim_lens.source_redshift_list[0]

        # extract SN metadata from the extended source's source_dict
        source_dict = slsim_lens._source[0]._source._extended_source.source_dict
        sn_type = source_dict.get('sn_type')
        lightcurve_time = source_dict.get('lightcurve_time')

        # determine which bands have SN light curve data
        kwargs_variability = source_dict.get('kwargs_variability', set())
        sn_bands = {b for b in kwargs_variability if b != 'supernovae_lightcurve'}

        # get bands from deflector if not provided
        if bands is None:
            bands = [k.split("_")[1] for k in
                     slsim_lens.deflector._deflector._deflector_dict.keys()
                     if k.startswith("mag_")]

        # filter to bands that the SN has light curve data for
        sn_light_curve_bands = [b for b in bands if b in sn_bands]

        # SLSim requires a time parameter for supernovae since there is no
        # static point source magnitude. Use t=0 as the reference epoch.
        ref_time = 0.0

        # get kwargs_model and kwargs_params (includes point source for PointPlusExtendedSource)
        kwargs_model, kwargs_params = slsim_lens.lenstronomy_kwargs(
            band=sn_light_curve_bands[0], time=ref_time
        )

        # collect band-specific source images (for catalog sources e.g. COSMOS_WEB)
        # only check bands the SN light curve model supports
        source_images = {}
        for band in sn_light_curve_bands:
            _, band_kwargs = slsim_lens.lenstronomy_kwargs(band=band, time=ref_time)
            if 'image' in band_kwargs['kwargs_source'][0]:
                source_images[band] = band_kwargs['kwargs_source'][0]['image']
        if source_images:
            kwargs_params['source_images'] = source_images

        # add additional necessary key/value pairs to kwargs_model
        kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_params['kwargs_lens'])
        kwargs_model['source_redshift_list'] = [z_source]
        kwargs_model['cosmo'] = cosmo
        kwargs_model['z_source'] = z_source

        # populate magnitudes dictionary (extended source = host galaxy)
        lens_mags, source_mags, lensed_source_mags = {}, {}, {}
        for band in bands:
            lens_mags[band] = slsim_lens.deflector_magnitude(band)
            source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=False)[0]
            lensed_source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=True)[0]
        magnitudes = {
            'lens': lens_mags,
            'source': source_mags,
            'lensed_source': lensed_source_mags,
        }

        # pre-compute light curves for each SN band
        light_curves = {}
        if lightcurve_time is not None:
            for band in sn_light_curve_bands:
                ps_mags = slsim_lens.point_source_magnitude(
                    band, lensed=True, time=lightcurve_time
                )
                # ps_mags is a list (per source); take first source
                # each element is a list of arrays (one per lensed image)
                light_curves[band] = {
                    'time': lightcurve_time,
                    'magnitudes': ps_mags[0],
                }

        # extract time delays and image magnifications
        time_delays = slsim_lens.point_source_arrival_times()
        image_magnifications = slsim_lens.point_source_magnification()

        # populate physical parameters
        physical_params = {
            'einstein_radius': slsim_lens.einstein_radius[0],
            'lens_stellar_mass': slsim_lens.deflector_stellar_mass(),
            'lens_velocity_dispersion': slsim_lens.deflector_velocity_dispersion(),
            'magnification': slsim_lens.extended_source_magnification[0],
            'magnitudes': magnitudes,
            'sn_type': sn_type,
            'time_delays': time_delays[0],  # first source
            'image_magnifications': image_magnifications[0],  # first source
            'light_curves': light_curves,
        }
        if slsim_lens.deflector.deflector_type == "NFW_HERNQUIST":
            physical_params['main_halo_mass'] = slsim_lens.deflector.halo_properties[0]
            physical_params['main_halo_concentration'] = slsim_lens.deflector.halo_properties[1]

        return LensedSupernova(name=name,
                               coords=coords,
                               kwargs_model=kwargs_model,
                               kwargs_params=kwargs_params,
                               physical_params=physical_params,
                               use_jax=use_jax)
