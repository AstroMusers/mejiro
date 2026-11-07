from astropy.units import Quantity
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST as LenstronomyLSST
from lenstronomy.Util import data_util

from mejiro.instruments.instrument import Instrument


class LSST(Instrument):
    """The Rubin Observatory's Legacy Survey of Space and Time (LSST).

    Instrument parameters come from lenstronomy's LSST observation config. The PSF is a Gaussian
    with the band's median seeing.
    """

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

        self.kwargs_single_band_dict = {band: LenstronomyLSST(band=band).kwargs_single_band() for band in bands}
        self.zeropoints = {band: kwargs['magnitude_zero_point'] for band, kwargs in self.kwargs_single_band_dict.items()}

        # the sky brightness is in mag / arcsec^2; convert it to counts / pixel / second
        self.sky_level = {
            band: Quantity(data_util.magnitude2cps(kwargs['sky_brightness'], magnitude_zero_point=self.zeropoints[band])
                           * self.get_pixel_scale(band).value ** 2, 'ct / pix')
            for band, kwargs in self.kwargs_single_band_dict.items()
        }
        # lenstronomy's sky brightness is the whole background, so there's no separate thermal term
        self.thermal_background = {band: Quantity(0.0, 'ct / pix') for band in bands}
        self.stray_light_fraction = 0.

    def get_gain(self, band):
        return self.kwargs_single_band_dict[band]['ccd_gain']

    def get_dark_current(self, band):
        # lenstronomy's LSST config has no dark current
        return Quantity(0.0, 'ct / (pix s)')

    def get_read_noise(self, band):
        return Quantity(self.kwargs_single_band_dict[band]['read_noise'], 'ct / pix')

    def get_pixel_scale(self, band):
        return Quantity(self.kwargs_single_band_dict[band]['pixel_scale'], 'arcsec / pix')

    def get_psf_fwhm(self, band):
        return Quantity(self.kwargs_single_band_dict[band]['seeing'], 'arcsec')

    def get_psf_kwargs(self, band, **kwargs):
        """A Gaussian PSF with the band's median seeing. Takes the PSF arguments the pipeline passes
        to every instrument, but uses none of them."""
        from mejiro.utils import lenstronomy_util
        return lenstronomy_util.get_gaussian_psf_kwargs(self.get_psf_fwhm(band))

    def get_sky_level(self, band):
        return self.sky_level[band]

    def get_thermal_background(self, band):
        return self.thermal_background[band]

    def get_zeropoint_magnitude(self, band):
        return self.zeropoints[band]

    @staticmethod
    def load_speclite_filters():
        from speclite.filters import load_filters
        return load_filters(*[f'lsst2023-{band}' for band in ['u', 'g', 'r', 'i', 'z', 'y']])

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
