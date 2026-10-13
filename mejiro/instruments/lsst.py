import importlib.metadata
import os
import numpy as np
from types import SimpleNamespace
import chalcedon
from chalcedon.imaging import data_util
from astropy import units as u
from astropy.units import Quantity
from fractions import Fraction
from glob import glob

from mejiro.instruments.instrument import Instrument


class LSST(Instrument):
    def __init__(self):
        name = 'LSST'
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        num_detectors = 1
        engines = ['galsim']

        super().__init__(
            name,
            bands,
            num_detectors,
            engines
        )

        self.versions['chalcedon'] = chalcedon.__version__
        
        # set attributes
        self.gain = 2.3
        self.stray_light_fraction = 0.
        self.pixel_scale = 0.2  # [arcsec / pixel]

        zeropoints = dict(zip(bands, (26.5, 28.3, 28.13, 27.79, 27.4, 26.58)))  # [mag]
        seeing = dict(zip(bands, (0.81, 0.77, 0.73, 0.71, 0.69, 0.68)))  # [arcsec]
        self.band_obs = {band: SimpleNamespace(obs={'seeing': seeing[band],
                                                   'magnitude_zero_point': zeropoints[band]}) for band in bands}
        self.lenstronomy_band_obs = self.band_obs

        self.sky_level = None
    
    def get_sky_level(self, band):
        return self.sky_level[band]

    def get_gain(self, band):
        return self.gain

    def get_noise(self, band):
        """
        Estimate noise per pixel per second in given band. For now, sum of dark current and read noise.
        """
        dark_current = self.get_dark_current(band)
        read_noise = self.get_read_noise(band)  # TODO the units aren't right for this sum to work

        return dark_current + read_noise

    def get_dark_current(self, band):
        return Quantity(0.0, 'ct / (pix s)')

    def get_read_noise(self, band):
        return Quantity(10.0, 'ct / (pix s)')
    
    # implement abstract methods
    def get_pixel_scale(self, band):
        return Quantity(0.2, 'arcsec / pix')
    
    def get_psf_fwhm(self, band):
        return self.band_obs[band].obs['seeing'] * u.Unit('arcsec')

    def get_psf_kwargs(self, band, **kwargs):
        from mejiro.utils import lenstronomy_util
        psf_fwhm = self.get_psf_fwhm(band)
        return lenstronomy_util.get_gaussian_psf_kwargs(psf_fwhm)
    
    def get_thermal_background(self, band):
        return Quantity(0.0, 'ct / (pix s)')
    
    def get_zeropoint_magnitude(self, band):
        return self.band_obs[band].obs['magnitude_zero_point']
    
    @staticmethod
    def load_speclite_filters():
        from speclite.filters import load_filters
        return load_filters('lsst2023-*')

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
