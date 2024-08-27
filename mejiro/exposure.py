import numpy as np
import galsim

from mejiro.helpers import gs, psf


class Exposure:

    def __init__(self, synthetic_image, exposure_time):
        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time

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
        image = galsim.Convolve([interp, self.psf])

        # TODO add noise based on instrument and synthetic image band

        # draw image
        im = galsim.ImageF(self.synthetic_image.num_pix, self.synthetic_image.num_pix)
        final = image.drawImage(im, scale=self.synthetic_image.native_pixel_scale)

        # set exposure
        self.exposure = final.array
