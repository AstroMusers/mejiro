import numpy as np
from astropy import units as u
from fractions import Fraction
from syotools.models import Camera, Telescope

from mejiro.instruments.instrument_base import InstrumentBase


class HWO(InstrumentBase):
    """
    Habitable Worlds Observatory (HWO) instrument class

    Parameters
    ----------
    eac : str
        Exploratory Analytic Case (EAC), possible mission architectures. Options are 'EAC1', 'EAC2', and 'EAC3'. Default is 'EAC1'. For more details, see https://pcos.gsfc.nasa.gov/physpag/meetings/HEAD2024/presentations/6_Burns_HWO.pdf
    """

    def __init__(self, eac='EAC1'):
        self.telescope = Telescope()
        self.camera = Camera()

        name = 'HWO'
        bands = self.camera.bandnames
        engines = ['galsim']

        super().__init__(
            name,
            bands,
            engines
        )

        # set aperture
        assert self.telescope.recover('aperture').unit == u.m, "Aperture must be in units of meters"  # check that aperture is in meters
        self.aperture = self.telescope.recover('aperture').value  # meters

        self.pivotwave = self._set_camera_attribute_from_hwo_tools('pivotwave', u.nm)
        self.ab_zeropoint = [35548., 24166., 15305., 12523., 10018., 8609., 6975., 4373., 3444., 2482.]
        self.aperture_correction = self._set_camera_attribute_from_hwo_tools('ap_corr')
        self.bandpass_r = self._set_camera_attribute_from_hwo_tools('bandpass_r')
        # self.derived_bandpass = {band: (self.pivotwave[band] / self.bandpass_r[band]) for band in self.bands}

        self.gain = 1.

        # set noise parameters
        self.dark_current = self._set_camera_attribute_from_hwo_tools('dark_current', u.electron / (u.pix * u.second))
        self.read_noise = self._set_camera_attribute_from_hwo_tools('detector_rn', u.electron ** Fraction(1, 2) / u.pix ** Fraction(1, 2))

        # private attributes
        self._pixel_size = np.array(
            [0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.04, 0.04, 0.04])  # set by aperture in method below

        # methods
        # self._set_pixel_scale()
        # self._set_psf_fwhm()

    def validate_instrument_params(self, params):
        # TODO implement this
        pass

    def _set_camera_attribute_from_hwo_tools(self, attribute, unit=None):
        if type(self.camera.recover(attribute)) == u.Quantity:
            values = self.camera.recover(attribute).value
        elif type(self.camera.recover(attribute)) == list:
            values = self.camera.recover(attribute)
        else:
            raise ValueError(f"Cannot recover {attribute} of type {type(self.camera.recover(attribute))}")

        assert len(self.bands) == len(values), f"Length of {attribute} does not match number of bands"
        if unit is not None:
            assert self.camera.recover(attribute).unit == unit, f"Unit of {attribute} must be {unit}"

        return {band: value for band, value in zip(self.bands, values)}

    def _set_pixel_scale(self):
        self.pixel_scale = 1.22 * (self.pivotwave * 0.000000001) * 206264.8062 / self.aperture / 2.
        # this enforces the rule that the pixel sizes are set at the shortest wavelength in each channel 
        self.pixel_scale[0:2] = 1.22 * (
                self.pivotwave[2] * 0.000000001) * 206264.8062 / self.aperture / 2.  # UV set at U
        self.pixel_scale[2:-3] = 1.22 * (
                self.pivotwave[2] * 0.000000001) * 206264.8062 / self.aperture / 2.  # Opt set at U
        self.pixel_scale[-3:] = 1.22 * (
                self.pivotwave[7] * 0.000000001) * 206264.8062 / self.aperture / 2.  # NIR set at J

    def _set_psf_fwhm(self):
        diff_limit = 1.22 * (500. * 0.000000001) * 206264.8062 / self.aperture

        self.fwhm_psf = 1.22 * self.pivotwave * 0.000000001 * 206264.8062 / self.aperture
        self.fwhm_psf[self.fwhm_psf < diff_limit] = self.fwhm_psf[self.fwhm_psf < diff_limit] * 0.0 + diff_limit

    def get_pixel_scale(self, band):
        index = self._get_index(band)
        return self._pixel_size[index]

    def get_psf_fwhm(self, band):
        index = self._get_index(band)
        return self.fwhm_psf[index]

    def get_zeropoint_magnitude(self, band):
        """
        Return zeropoint AB magnitude in given band

        Source: https://jt-astro.science/luvoir_simtools/hdi_etc/SNR_equation.pdf
        """
        index = self._get_index(band)

        # get telescope aperture diameter in cm
        aperture = self.aperture * 100

        # get band-specific parameters
        bandwidth = self.derived_bandpass[index]  # TODO is this correct?
        flux_zp = self.ab_zeropoint[index]

        # calculate zeropoint magnitude
        zp_mag = (-1 / 0.4) * np.log10(4 / (np.pi * flux_zp * (aperture ** 2) * bandwidth))

        return zp_mag

    def get_noise(self, band):
        """
        Estimate noise per pixel per second in given band. For now, sum of dark current and read noise.
        """
        index = self._get_index(band)

        dark_current = self.dark_current[index]
        read_noise = self.read_noise[index]

        return dark_current + read_noise

    def get_dark_current(self, band):
        index = self._get_index(band)
        return self.dark_current[index]

    def get_read_noise(self, band):
        index = self._get_index(band)
        return self.read_noise[index]

    def get_bands(self):
        return self.camera.recover('bandnames')

    def _get_index(self, band):
        """
        hwo-tools provides parameters in lists, so to get value for a particular band, we need the index of that band.
        """
        # handle if provided in lower case
        band = band.upper()

        # check if band is in camera bandnames
        # bands = self.camera.bandnames
        bands = self.bands
        if band not in bands:
            raise ValueError(f"Band {band} not in {bands}")

        return bands.index(band)

    @staticmethod
    def default_params():
        return {}

    @staticmethod
    def validate_instrument_params(params):
        return params
