import os
from glob import glob

from astropy.units import Quantity

from mejiro.instruments.instrument import Instrument

# AB zeropoints of WFC3/UVIS1 for a 6" (effectively infinite) aperture on MJD 60035
# (2023-04-01), computed once with stsynphot 1.5.0 from the `wfc3,uvis1,<band>,mjd#60035,aper#6.0`
# obsmode as ST magnitude (-21.1 - 2.5 log10 PHOTFLAM) minus 5 log10(pivot wavelength in
# Angstrom) plus 18.6921. They're stored here so mejiro doesn't need stsynphot and its STScI
# throughput tables at runtime.
ZEROPOINTS = {
    'F438W': 24.8159,
    'F475W': 25.6839,
    'F606W': 26.0575,
    'F814W': 25.1097,
}

# https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-6-uvis-optical-performance
PSF_FWHM = {
    'F438W': 0.070,
    'F475W': 0.067,
    'F606W': 0.067,
    'F814W': 0.074,
}


class HST(Instrument):
    """The Hubble Space Telescope's WFC3/UVIS camera.

    The PSF is a Gaussian with the band's FWHM. The sky level is a placeholder.
    """

    def __init__(self):
        name = 'HST'
        bands = list(ZEROPOINTS)
        num_detectors = 1
        engines = ['galsim']

        super().__init__(
            name,
            bands,
            num_detectors,
            engines
        )

        self.gain = 2.5  # e- / DN
        self.stray_light_fraction = 0.1
        # https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html
        self.dark_current = Quantity(0.00319, 'ct / (pix s)')
        # https://ntrs.nasa.gov/api/citations/20060047835/downloads/20060047835.pdf
        self.read_noise = Quantity(3.0, 'ct / pix')
        # "The thermal background is negligible below about 8000 Angstrom"
        # (https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html)
        self.thermal_background = {band: Quantity(0.0, 'ct / pix') for band in self.bands}
        # placeholder: a rough estimate, not from an HST reference
        self.sky_level = {band: Quantity(0.25, 'ct / pix') for band in self.bands}

    def get_gain(self, band):
        return self.gain

    def get_dark_current(self, band):
        return self.dark_current

    def get_read_noise(self, band):
        return self.read_noise

    def get_pixel_scale(self, band):
        return Quantity(0.04, 'arcsec / pix')

    def get_psf_fwhm(self, band):
        return Quantity(PSF_FWHM[band], 'arcsec')

    def get_psf_kwargs(self, band, **kwargs):
        """A Gaussian PSF with the band's FWHM. Takes the PSF arguments the pipeline passes to every
        instrument, but uses none of them."""
        from mejiro.utils import lenstronomy_util
        return lenstronomy_util.get_gaussian_psf_kwargs(self.get_psf_fwhm(band))

    def get_sky_level(self, band):
        return self.sky_level[band]

    def get_thermal_background(self, band):
        return self.thermal_background[band]

    def get_zeropoint_magnitude(self, band):
        return ZEROPOINTS[band]

    @staticmethod
    def load_speclite_filters():
        import mejiro
        module_path = os.path.dirname(mejiro.__file__)
        filters = sorted(glob(os.path.join(module_path, 'data', 'hst_filter_response', 'WFC3_UVIS-*.ecsv')))

        from speclite.filters import load_filters
        return load_filters(*filters)

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
