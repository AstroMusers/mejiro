import os
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from fractions import Fraction
from glob import glob
from lenstronomy.SimulationAPI.ObservationConfig import JWST as LenstronomyJWST
from lenstronomy.Util import data_util

from mejiro.instruments.instrument import Instrument


class JWST(Instrument):
    def __init__(self):
        name = 'JWST'
        bands = ['F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
        num_detectors = 1
        engines = ['galsim']

        super().__init__(
            name,
            bands,
            num_detectors,
            engines
        )

        # set kwargs_single_band
        self.kwargs_single_band_dict = {}
        for band in bands:
            self.kwargs_single_band_dict[band] = LenstronomyJWST.JWST(band=band).kwargs_single_band()

        # set zeropoints
        self.zeropoints = {band: self.kwargs_single_band_dict[band]['magnitude_zero_point'] for band in bands}

        # set thermal background to zero because it's (ostensibly) included in lenstronomy sky brightness
        self.thermal_background = {band: 0 * u.ct / u.pix for band in self.bands}

        # calculate sky level
        sky_level_mag_per_arcsec2 = {band: self.kwargs_single_band_dict[band]['sky_brightness'] for band in bands}
        self.sky_level = {band: data_util.magnitude2cps(mag_per_arcsec2, magnitude_zero_point=self.get_zeropoint_magnitude(band)) * (self.get_pixel_scale(band) ** 2).value * u.Unit('ct / pix') for band, mag_per_arcsec2 in sky_level_mag_per_arcsec2.items()}

        # set attributes
        self.stray_light_fraction = 0.1

    def get_gain(self, band):
        return self.kwargs_single_band_dict[band]['ccd_gain']
    
    def get_read_noise(self, band):
        read_noise = self.kwargs_single_band_dict[band]['read_noise']
        return Quantity(read_noise, u.electron ** Fraction(1, 2) / u.pix ** Fraction(1, 2))
    
    def get_dark_current(self, band):
        if band in ['F115W', 'F150W', 'F200W']:
            dark_current = 1.9 * 1e-3 * u.electron / (u.pix * u.second)
        else:
            dark_current = 34.2 * 1e-3 * u.electron / (u.pix * u.second)
        return dark_current

    def get_pixel_scale(self, band):
        pixel_scale = self.kwargs_single_band_dict[band]['pixel_scale']
        return Quantity(pixel_scale, u.arcsec / u.pix)

    def get_psf_kwargs(self, band, **kwargs):
        from mejiro.engines.stpsf_engine import STPSFEngine
        return STPSFEngine.get_jwst_psf_kwargs(band, kwargs['oversample'], kwargs['num_pix'], kwargs.get('check_cache', False), kwargs.get('psf_cache_dir', None), kwargs.get('verbose', False))

    def get_psf_fwhm(self, band):
        pass

    def get_sky_level(self, band):
        return self.sky_level[band]
    
    def get_thermal_background(self, band):
        return self.thermal_background[band]

    def get_zeropoint_magnitude(self, band):
        return self.zeropoints[band]

    @staticmethod
    def load_speclite_filters():
        import mejiro
        module_path = os.path.dirname(mejiro.__file__)
        filters = sorted(glob(os.path.join(module_path, 'data', 'jwst_filter_response', f'NIRCam-*.ecsv')))

        from speclite.filters import load_filters
        return load_filters(*filters)
    
    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
