import importlib.metadata
import os
import numpy as np
from chalcedon.imaging import data_util
import stsynphot as stsyn
from astropy import units as u
from astropy.units import Quantity
from fractions import Fraction
from glob import glob

from mejiro.instruments.instrument import Instrument


class HST(Instrument):
    def __init__(self):
        name = 'HST'
        bands = ['F438W', 'F475W', 'F606W', 'F814W']
        num_detectors = 1
        engines = ['galsim']

        super().__init__(
            name,
            bands,
            num_detectors,
            engines
        )

        detectors = ['uvis1']  # , 'uvis2'
        filtnames = ['F438W', 'F475W', 'F555W', 'F606W', 'F625W', 'F775W', 'F814W']
        mjd = '60035'  # April 1st, 2023
        # mjd = str(Time.now().mjd)  # Time right now

        #Aperture Radius
        aper = '6.0'  # 151 pixels (infinity) [default behavior]
        #aper = '0.396'              # 10 pixels for UVIS

        # set attributes
        self.gain = 2.5
        self.stray_light_fraction = None
        self.dark_current = {band: Quantity(0.00319, 'ct / (pix s)') for band in filtnames}  # https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html
        self.read_noise = {band: Quantity(3.0, 'ct / (pix s)') for band in filtnames}  # https://ntrs.nasa.gov/api/citations/20060047835/downloads/20060047835.pdf
        self.psf_fwhm = {
            'F438W': Quantity(0.070, 'arcsec'),
            'F475W': Quantity(0.067, 'arcsec'),
            'F555W': Quantity(0.067, 'arcsec'),
            'F606W': Quantity(0.067, 'arcsec'),
            'F625W': Quantity(0.067, 'arcsec'),
            'F775W': Quantity(0.074, 'arcsec'),
            'F814W': Quantity(0.074, 'arcsec')
        }  # https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-6-uvis-optical-performance
        self.thermal_background = {
            'F438W': Quantity(0.0, 'ct / pix'),
            'F475W': Quantity(0.0, 'ct / pix'),
            'F555W': Quantity(0.0, 'ct / pix'),
            'F606W': Quantity(0.0, 'ct / pix'),
            'F625W': Quantity(0.0, 'ct / pix'),
            'F775W': Quantity(0.0, 'ct / pix'),
            'F814W': Quantity(0.0, 'ct / pix')
        }  # "The thermal background is negligible below about 8000 Angstrom" (https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html)
        self.sky_level = {band: None for band in filtnames}

    @staticmethod
    def build_obsmode(detector, filt, mjd, aper):
        return f'wfc3,{detector},{filt},mjd#{mjd},aper#{aper}'
    
    @staticmethod
    def calculate_zp_abmag(bp):
        # STMag
        photflam = bp.unit_response(stsyn.conf.area)  # inverse sensitivity in flam
        stmag = -21.1 - 2.5 * np.log10(photflam.value)

        # Pivot Wavelength and bandwidth
        photplam = bp.pivot()  # pivot wavelength in angstroms
        bandwidth = bp.photbw()  # bandwidth in angstroms

        # ABMag
        return stmag - 5 * np.log10(photplam.value) + 18.6921
    
    def get_sky_level(self, band):
        sky_level = self.sky_level[band]
        if sky_level is None:
            raise NotImplementedError(f'HST sky level is not configured for band {band}')
        return sky_level

    def get_gain(self, band):
        return self.gain

    def get_dark_current(self, band):
        return self.dark_current[band]

    def get_read_noise(self, band):
        return self.read_noise[band]
    
    # implement abstract methods
    def get_pixel_scale(self, band):
        return Quantity(0.04, 'arcsec / pix')
    
    def get_psf_fwhm(self, band):
        return self.psf_fwhm[band]

    def get_thermal_background(self, band):
        return self.thermal_background[band]
    
    def get_zeropoint_magnitude(self, band):
        band = band.lower()
        obsmode = HST.build_obsmode('uvis1', band, '60035', '6.0')
        bp = stsyn.band(obsmode)
        return HST.calculate_zp_abmag(bp)
    
    @staticmethod
    def load_speclite_filters():
        import mejiro
        module_path = os.path.dirname(mejiro.__file__)
        filters = sorted(glob(os.path.join(module_path, 'data', 'hst_filter_response', f'WFC3_UVIS-*.ecsv')))

        from speclite.filters import load_filters
        return load_filters(*filters)

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
