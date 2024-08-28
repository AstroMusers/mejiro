import galsim
import numpy as np
import math

from mejiro.instruments.instrument_base import InstrumentBase
from mejiro.helpers import psf

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

    def get_exposure(self, synthetic_image, interp, rng, exposure_time):
        # get PSF
        self.psf_fwhm = self.get_psf_fwhm(self.synthetic_image.band)
        print(f'PSF FWHM: {self.psf_fwhm}')
        # self.psf = psf.get_gaussian_psf(self.psf_fwhm, self.synthetic_image.oversample,
                                        # pixel_scale=self.synthetic_image.native_pixel_scale)
        self.psf = psf.get_gaussian_psf(self.psf_fwhm)
        self.psf_image = self.psf.drawImage(scale=self.synthetic_image.pixel_scale)

        # convolve with PSF
        convolved = galsim.Convolve([interp, self.psf])

        # draw image
        output_num_pix = math.floor(synthetic_image.num_pix / synthetic_image.oversample)
        im = galsim.ImageF(output_num_pix, output_num_pix, scale=synthetic_image.native_pixel_scale)
        im.setOrigin(0, 0)
        image = convolved.drawImage(im)

        # quantize, since integer number of photo-electrons are being created
        image.quantize()

        # add sky background
        sky_level = None 
        # TODO remember to use output_num_pix

        # add Poisson noise due to arrival times of photons from signal and sky
        poisson_noise = galsim.PoissonNoise(rng)
        image.addNoise(poisson_noise)

        # add dark current
        dark_current = self.get_dark_current(self.synthetic_image.band)
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(rng, dark_current))
        image.addNoise(dark_noise)

        # add read noise
        read_noise_sigma = self.get_read_noise(self.synthetic_image.band)
        read_noise = galsim.GaussianNoise(rng, sigma=read_noise_sigma)
        image.addNoise(read_noise)

        # gain
        image /= self.gain

        # quantize, since analog-to-digital conversion gives integers
        image.quantize()


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
