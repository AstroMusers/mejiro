import math
from copy import deepcopy

import galsim
import numpy as np

from mejiro.helpers import psf
from mejiro.instruments.instrument_base import InstrumentBase


# from syotools.models import Camera, Telescope


class HWO(InstrumentBase):

    def __init__(self, aperture=10.):
        name = 'HWO'

        super().__init__(
            name
        )

        # TODO eventually work with their ordereddict and astropy quantity stuff
        # self.telescope = Telescope()
        # self.camera = Camera()

        self.aperture = aperture  # meters 
        self.dark_current = [0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002]
        self.read_noise = [3., 3., 3., 3., 3., 3., 3., 4., 4., 4.]
        self.pivotwave = np.array([155., 228., 360., 440., 550., 640., 790., 1260., 1600., 2220.])
        self.bands = ['FUV', 'NUV', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
        self.ab_zeropoint = [35548., 24166., 15305., 12523., 10018., 8609., 6975., 4373., 3444., 2482.]
        self.aperture_correction = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        self.bandpass_r = [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]
        self.derived_bandpass = [pw / bp_r for pw, bp_r in zip(self.pivotwave, self.bandpass_r)]
        self.gain = 1.

        # private attributes
        self._pixel_size = np.array(
            [0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.04, 0.04, 0.04])  # set by aperture in method below

        # methods
        self._set_pixel_scale()
        self._set_psf_fwhm()

    def get_exposure(self, synthetic_image, interp, rng, exposure_time, sky_background=True, detector_effects=True, **kwargs):
        suppress_output = kwargs['suppress_output'] if 'suppress_output' in kwargs else True

        # get PSF
        self.psf_fwhm = self.get_psf_fwhm(synthetic_image.band)
        if not suppress_output: print(f'PSF FWHM: {self.psf_fwhm}')
        # self.psf = psf.get_gaussian_psf(self.psf_fwhm, self.synthetic_image.oversample,
        # pixel_scale=self.synthetic_image.native_pixel_scale)
        self.psf = psf.get_gaussian_psf(self.psf_fwhm)
        self.psf_image = self.psf.drawImage(scale=synthetic_image.pixel_scale)

        # convolve with PSF
        convolved = galsim.Convolve([interp, self.psf])

        # draw image
        output_num_pix = math.floor(synthetic_image.num_pix / synthetic_image.oversample)
        im = galsim.ImageF(output_num_pix, output_num_pix, scale=synthetic_image.native_pixel_scale)
        im.setOrigin(0, 0)
        image = convolved.drawImage(im)

        # NB from here on out, the image is at the native pixel scale, i.e., the image is NOT oversampled

        # add sky background
        if sky_background:
            min_zodi_cps = 0.2
            sky_bkg_cps = min_zodi_cps * 1.5

            # build Image
            sky_image = galsim.ImageF(output_num_pix, output_num_pix)
            sky_image += sky_bkg_cps

            # convert to counts/pixel
            sky_image *= exposure_time

            image += sky_image
        
        if detector_effects:
            image.replaceNegative(0.)

            # add Poisson noise due to arrival times of photons from signal and sky
            if 'poisson_noise' not in kwargs:
                before = deepcopy(image)
                image.addNoise(galsim.PoissonNoise(rng))
                poisson_noise = image - before
                image.quantize()
            elif type(kwargs['poisson_noise']) is galsim.Image:
                image += kwargs['poisson_noise']
                image.quantize()
            elif type(kwargs['poisson_noise']) is bool:
                poisson_noise = None
                pass

            # dark current
            if 'dark_noise' not in kwargs:
                before = deepcopy(image)
                total_dark_current = self.get_dark_current(synthetic_image.band)
                image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, total_dark_current)))
                dark_noise = image - before
            elif type(kwargs['dark_noise']) is galsim.Image:
                image += kwargs['dark_noise']
            elif type(kwargs['dark_noise']) is bool:
                dark_noise = None
                pass

            # read noise
            if 'read_noise' not in kwargs:
                before = deepcopy(image)
                read_noise_sigma = self.get_read_noise(synthetic_image.band)
                image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
                read_noise = image - before
            elif type(kwargs['read_noise']) is galsim.Image:
                image += kwargs['read_noise']
            elif type(kwargs['read_noise']) is bool:
                read_noise = None
                pass

        # gain
        image /= self.gain

        # quantize, since analog-to-digital conversion gives integers
        image.quantize()

        if detector_effects and 'return_noise' in kwargs and kwargs['return_noise']:
            return image, poisson_noise, dark_noise, read_noise
        else:
            return image

    @staticmethod
    def validate_instrument_config(config):
        # TODO implement this
        pass

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
