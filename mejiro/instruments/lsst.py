import importlib.metadata
import os
import numpy as np
import lenstronomy.Util.data_util as data_util
from astropy import units as u
from astropy.units import Quantity
from fractions import Fraction
from glob import glob
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST as LenstronomyLSST

from mejiro.instruments.instrument import Instrument


class LSST(Instrument):
    def __init__(self):
        name = 'LSST'
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        num_detectors = 1
        engines = ['lenstronomy']

        super().__init__(
            name,
            bands,
            num_detectors,
            engines
        )

        # record versions of dependencies
        try:
            self.versions['lenstronomy'] = importlib.metadata.version('lenstronomy')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError("lenstronomy package not found. Please install lenstronomy.")
        
        lenstronomy_lsst = LenstronomyLSST()
        
        # set attributes
        self.gain = lenstronomy_lsst.camera['ccd_gain']
        self.stray_light_fraction = 0.
        self.pixel_scale = lenstronomy_lsst.camera['pixel_scale']

        self.lenstronomy_band_obs = {
            'u': LenstronomyLSST(band='u', psf_type='GAUSSIAN', coadd_years=10),
            'g': LenstronomyLSST(band='g', psf_type='GAUSSIAN', coadd_years=10),
            'r': LenstronomyLSST(band='r', psf_type='GAUSSIAN', coadd_years=10),
            'i': LenstronomyLSST(band='i', psf_type='GAUSSIAN', coadd_years=10),
            'z': LenstronomyLSST(band='z', psf_type='GAUSSIAN', coadd_years=10),
            'y': LenstronomyLSST(band='y', psf_type='GAUSSIAN', coadd_years=10),
        }

        self.sky_level = None
    
    def get_sky_level(self, band):
        return self.sky_level[band]

    def get_noise(self, band):
        """
        Estimate noise per pixel per second in given band. For now, sum of dark current and read noise.
        """
        dark_current = self.dark_current[band]
        read_noise = self.read_noise[band]  # TODO the units aren't right for this sum to work

        return dark_current + read_noise

    def get_dark_current(self, band):
        return Quantity(0.0, 'ct / pix / s')

    def get_read_noise(self, band):
        return Quantity(10.0, 'ct / pix / s')
    
    # implement abstract methods
    def get_pixel_scale(self, band):
        return Quantity(0.2, 'arcsec / pix')
    
    def get_psf_fwhm(self, band):
        return self.lenstronomy_band_obs[band]['seeing'] * u.Unit('arcsec')

    def get_psf_kwargs(self, band, **kwargs):
        from mejiro.utils import lenstronomy_util
        psf_fwhm = self.get_psf_fwhm(band)
        return lenstronomy_util.get_gaussian_psf_kwargs(psf_fwhm)
    
    def get_thermal_background(self, band):
        return Quantity(0.0, 'ct / pix / s')
    
    def get_zeropoint_magnitude(self, band):
        return self.lenstronomy_band_obs[band].obs['magnitude_zero_point']
    
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
