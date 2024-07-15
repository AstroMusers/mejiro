import numpy as np
from syotools.models import Camera, Telescope

from mejiro.instruments.instrument_base import InstrumentBase


class HWO(InstrumentBase):

    def __init__(self):
        # TODO eventually work with their ordereddict and astropy quantity stuff
        # self.telescope = Telescope()
        self.camera = Camera()

        self.aperture = 10.  # aperture in meters 
        self.dark_current = [0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002]
        self.read_noise = [3., 3., 3., 3., 3., 3., 3., 4., 4., 4.]
        self.pivotwave = [155., 228., 360., 440., 550., 640., 790., 1260., 1600., 2220.]
        self.bandnames = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
        self.ab_zeropoint = [35548., 24166., 15305., 12523., 10018., 8609., 6975., 4373., 3444., 2482.]
        self.aperture_correction = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        self.bandpass_r = [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]
        self.derived_bandpass = [pw / bp_r for pw, bp_r in zip(self.pivotwave, self.bandpass_r)]

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
        bands = self.camera.bandnames
        if band not in bands:
            raise ValueError(f"Band {band} not in {bands}")

        return bands.index(band)
