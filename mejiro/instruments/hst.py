import importlib.metadata
import os
import numpy as np
import lenstronomy.Util.data_util as data_util
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

        # record versions of dependencies
        try:
            self.versions['lenstronomy'] = importlib.metadata.version('lenstronomy')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError("lenstronomy package not found. Please install lenstronomy.")
        
        detectors = ['uvis1']  # , 'uvis2'
        filtnames = ['F438W', 'F475W', 'F555W', 'F606W', 'F625W', 'F775W', 'F814W']
        mjd = '60035'  # April 1st, 2023
        # mjd = str(Time.now().mjd)  # Time right now

        #Aperture Radius
        aper = '6.0'  # 151 pixels (infinity) [default behavior]
        #aper = '0.396'              # 10 pixels for UVIS

        # set attributes
        self.gain = 2.5
        self.stray_light_fraction = 0.1
        self.pixel_scale = None
        self.dark_current = {
            'F160W': Quantity(0.0, 'ct / pix / s'),  # TODO: placeholder value, needs to be updated
        }
        self.read_noise = {
            'F160W': Quantity(4.0, 'ct / pix / s'),
        }
        self.psf_fwhm = {
            'F160W': Quantity(0.08, 'arcsec'),
        }
        self.thermal_background = {
            'B': Quantity(0.0, 'ct / pix'),  # TODO: placeholder value, needs to be updated
        }
        self.zeropoints = {
            'F160W': 25.96,
        }
        self.sky_level = None

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
        return self.sky_level[band]

    def get_noise(self, band):
        """
        Estimate noise per pixel per second in given band. For now, sum of dark current and read noise.
        """
        dark_current = self.dark_current[band]
        read_noise = self.read_noise[band]  # TODO the units aren't right for this sum to work

        return dark_current + read_noise

    def get_dark_current(self, band):
        return self.dark_current[band]

    def get_read_noise(self, band):
        return self.read_noise[band]
    
    # implement abstract methods
    def get_pixel_scale(self, band):
        return Quantity(0.08, 'arcsec / pix')
    
    def get_psf_fwhm(self, band):
        return self.psf_fwhm[band]

    def get_psf_kwargs(self, band, **kwargs):
        from mejiro.utils import lenstronomy_util
        psf_fwhm = self.get_psf_fwhm(band)
        return lenstronomy_util.get_gaussian_psf_kwargs(psf_fwhm)
    
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
        filters = sorted(glob(os.path.join(module_path, 'data', 'hwo_filter_response', f'HRI-*.ecsv')))

        from speclite.filters import load_filters
        return load_filters(*filters)

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
