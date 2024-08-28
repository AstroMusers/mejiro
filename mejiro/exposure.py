import numpy as np
import galsim

from mejiro.helpers import gs, psf


class Exposure:

    def __init__(self, synthetic_image, exposure_time, seed=1):
        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time

        # set GalSim random number generator
        self.rng = galsim.UniformDeviate(seed)

        # get PSF
        self.psf_fwhm = self.synthetic_image.instrument.get_psf_fwhm(self.synthetic_image.band)
        print(f'PSF FWHM: {self.psf_fwhm}')
        # self.psf = psf.get_gaussian_psf(self.psf_fwhm, self.synthetic_image.oversample,
                                        # pixel_scale=self.synthetic_image.native_pixel_scale)
        self.psf = psf.get_gaussian_psf(self.psf_fwhm)
        self.psf_image = self.psf.drawImage(scale=self.synthetic_image.pixel_scale)

        # total flux cps
        self.lens_flux_cps = np.sum(self.synthetic_image.strong_lens.lens_light_model_class.total_flux(
            [self.synthetic_image.strong_lens.kwargs_lens_light_amp_dict[self.synthetic_image.band]]))
        self.source_flux_cps = np.sum(self.synthetic_image.strong_lens.source_model_class.total_flux(
            [self.synthetic_image.strong_lens.kwargs_source_amp_dict[self.synthetic_image.band]]))
        self.total_flux_cps = self.lens_flux_cps + self.source_flux_cps

        # create interpolated image
        interp = galsim.InterpolatedImage(galsim.Image(self.synthetic_image.image, xmin=0, ymin=0),
                                   scale=self.synthetic_image.pixel_scale,
                                   flux=self.total_flux_cps * self.exposure_time)

        # convolve with PSF
        convolved = galsim.Convolve([interp, self.psf])

        # draw image
        im = galsim.ImageF(self.synthetic_image.num_pix, self.synthetic_image.num_pix)
        image = convolved.drawImage(im, scale=self.synthetic_image.native_pixel_scale)

        # quantize, since integer number of photo-electrons are being created
        image.quantize()

        # add sky background
        sky_level = None

        # add Poisson noise due to arrival times of photons from signal and sky
        poisson_noise = galsim.PoissonNoise(self.rng)
        image.addNoise(poisson_noise)

        # add dark current
        dark_current = self.synthetic_image.instrument.get_dark_current(self.synthetic_image.band)
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, dark_current))
        image.addNoise(dark_noise)

        # add read noise
        read_noise_sigma = self.synthetic_image.instrument.get_read_noise(self.synthetic_image.band)
        read_noise = galsim.GaussianNoise(self.rng, sigma=read_noise_sigma)
        image.addNoise(read_noise)

        # gain
        image /= self.synthetic_image.instrument.gain

        # quantize, since analog-to-digital conversion gives integers
        image.quantize()

        # set exposure
        self.exposure = image.array
