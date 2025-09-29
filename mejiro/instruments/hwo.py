import importlib.metadata
import os
import numpy as np
import syotools
from lenstronomy.Util import data_util
from astropy import units as u
from astropy.units import Quantity
from fractions import Fraction
from syotools.models import Camera, Telescope
from glob import glob

from mejiro.instruments.instrument import Instrument


class HWO(Instrument):
    """
    Habitable Worlds Observatory (HWO) High-Resolution Imager (HRI) class

    Parameters
    ----------
    eac : str
        Exploratory Analytic Case (EAC), possible mission architectures. Options are 'EAC1', 'EAC2', and 'EAC3'. Default is 'EAC1'. For more details, see https://pcos.gsfc.nasa.gov/physpag/meetings/HEAD2024/presentations/6_Burns_HWO.pdf.

    Attributes
    ----------
    TODO
    """
    def __init__(self, eac='EAC1'):
        self.telescope = Telescope()
        self.telescope.set_from_json(eac)
        self.camera = Camera()
        self.telescope.add_camera(self.camera)

        name = 'HWO'
        bands = self.camera.bandnames
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
            self.versions['syotools'] = importlib.metadata.version('syotools')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError("syotools package not found. Please install syotools.")

        # set attributes
        self.gain = {band: 1. for band in self.bands}
        self.stray_light_fraction = 0.1
        self.aperture = self.telescope.recover('aperture')
        self.effective_aperture = self.telescope.effective_aperture
        self.pixel_scale = self.get_attribute_from_syotools(self.camera, 'pixel_size', u.arcsec / u.pix)
        self.dark_current = self.get_attribute_from_syotools(self.camera, 'dark_current', u.electron / (u.pix * u.second))
        self.read_noise = self.get_attribute_from_syotools(self.camera, 'detector_rn', u.electron ** Fraction(1, 2) / u.pix ** Fraction(1, 2))
        self.psf_fwhm = self.get_attribute_from_syotools(self.camera, 'fwhm_psf', u.arcsec, check_unit=False)  # TODO temporary override of unit check
        self.thermal_background = {
            'B': Quantity(0.0, 'ct / pix'),
            'FUV': Quantity(0.0, 'ct / pix'),
            'H': Quantity(0.0, 'ct / pix'),
            'I': Quantity(0.0, 'ct / pix'),
            'J': Quantity(0.0, 'ct / pix'),
            'K': Quantity(0.0, 'ct / pix'),
            'NUV': Quantity(0.0, 'ct / pix'),
            'R': Quantity(0.0, 'ct / pix'),
            'U': Quantity(0.0, 'ct / pix'),
            'V': Quantity(0.0, 'ct / pix'),
        }   # TODO these aren't real values, just placeholders

        # calculate zeropoints: https://jt-astro.science/luvoir_simtools/hdi_etc/SNR_equation.pdf
        aperture_cm = self.aperture.to(u.cm).value  # get telescope aperture diameter in cm
        bandwidth = {band: bp for band, bp in zip(self.bands, self.camera.recover('derived_bandpass'))}
        flux_zp = {band: bp for band, bp in zip(self.bands, self.camera.recover('ab_zeropoint'))}
        self.zeropoints = {band: (-1 / 0.4) * np.log10(4 / (np.pi * flux_zp[band].value * (aperture_cm ** 2) * bandwidth[band].value)) for band in self.bands}

        # calculate sky level
        sky_level_mag_per_arcsec2 = {band: value for band, value in self.get_attribute_from_syotools(self.camera, 'sky_sigma', None).items()}
        self.sky_level = {band: data_util.magnitude2cps(mag_per_arcsec2, magnitude_zero_point=self.get_zeropoint_magnitude(band)) * (self.get_pixel_scale(band) ** 2).value * u.Unit('ct / pix') for band, mag_per_arcsec2 in sky_level_mag_per_arcsec2.items()}


    def get_attribute_from_syotools(self, syotools_object, attribute, unit, check_unit=True):
        values = syotools_object.recover(attribute)
        
        if type(values) == list:
            # some syotools attributes are formatted as lists
            values = Quantity(values[0], values[1])

        # make sure that there's a value for each band
        if len(self.bands) != len(values):
            raise ValueError(f"Length of {attribute} does not match number of bands")

        # check units
        if check_unit:
            if type(values) == u.Quantity and unit is not None and values.unit != unit:
                raise ValueError(f"Unit of {attribute} must be {unit}, not {values.unit}")

        return {band: value for band, value in zip(self.bands, values)}
    
    def get_gain(self, band):
        return self.gain[band]
    
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
        return self.pixel_scale[band]
    
    def get_psf_fwhm(self, band):
        return self.psf_fwhm[band]

    def get_psf_kwargs(self, band, **kwargs):
        from mejiro.utils import lenstronomy_util
        psf_fwhm = self.get_psf_fwhm(band)
        return lenstronomy_util.get_gaussian_psf_kwargs(psf_fwhm)
    
    def get_thermal_background(self, band):
        return self.thermal_background[band]
    
    def get_zeropoint_magnitude(self, band):
        return self.zeropoints[band]
    
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
