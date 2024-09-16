import galsim
import numpy as np

from mejiro.utils import util


class Exposure:

    def __init__(self, synthetic_image, exposure_time, rng=None, **kwargs):
        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time

        # TODO validate instrument config
        # self.instrument.validate_instrument_config(config)
        # self.config = config

        # set GalSim random number generator
        if rng is not None:
            self.rng = rng
        else:
            self.rng = galsim.UniformDeviate(42)

        # total flux cps
        self.lens_flux_cps = self.synthetic_image.strong_lens.get_lens_flux_cps(self.synthetic_image.band,
                                                                                self.synthetic_image.magnitude_zero_point)
        self.source_flux_cps = self.synthetic_image.strong_lens.get_source_flux_cps(self.synthetic_image.band,
                                                                                    self.synthetic_image.magnitude_zero_point)
        self.total_flux_cps = self.lens_flux_cps + self.source_flux_cps

        # create interpolated image
        self.interp_total = galsim.InterpolatedImage(galsim.Image(self.synthetic_image.image, xmin=0, ymin=0),
                                                     scale=self.synthetic_image.pixel_scale,
                                                     flux=self.total_flux_cps * self.exposure_time)

        tuple = self.synthetic_image.instrument.get_exposure(self.synthetic_image, 
                                                             self.interp_total, self.rng,
                                                             self.exposure_time, **kwargs)
        if 'return_noise' in kwargs and kwargs['return_noise']:
            if self.synthetic_image.instrument.name == 'Roman':
                self.image, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = tuple
            elif self.synthetic_image.instrument.name == 'HWO':
                self.image, self.poisson_noise, self.dark_noise, self.read_noise = tuple
        else:
            self.image = tuple
        final = self.image.array

        # crop off edge effects (e.g., IPC)
        final_num_pix = final.shape[0]
        assert final_num_pix % 2 != 0, 'Final image has even number of pixels'
        output_num_pix = final_num_pix - 3
        final = util.center_crop_image(final, (output_num_pix, output_num_pix))

        # set exposure
        if np.any(final < 0):
            raise ValueError('Negative pixel values in final image')
        self.exposure = final

        if self.synthetic_image.pieces:
            self.lens_interp = galsim.InterpolatedImage(
                galsim.Image(self.synthetic_image.lens_surface_brightness, xmin=0, ymin=0),
                scale=self.synthetic_image.pixel_scale,
                flux=self.lens_flux_cps * self.exposure_time)
            self.source_interp = galsim.InterpolatedImage(
                galsim.Image(self.synthetic_image.source_surface_brightness, xmin=0, ymin=0),
                scale=self.synthetic_image.pixel_scale,
                flux=self.source_flux_cps * self.exposure_time)
            self.lens_image = self.synthetic_image.instrument.get_exposure(self.synthetic_image, self.lens_interp,
                                                                           self.rng, self.exposure_time,
                                                                           sky_background=False, detector_effects=False,
                                                                           **kwargs)
            self.source_image = self.synthetic_image.instrument.get_exposure(self.synthetic_image, self.source_interp,
                                                                             self.rng, self.exposure_time,
                                                                             sky_background=False,
                                                                             detector_effects=False, **kwargs)
            lens = self.lens_image.array
            source = self.source_image.array

            # crop off edge effects (e.g., IPC)
            lens = util.center_crop_image(lens, (output_num_pix, output_num_pix))
            source = util.center_crop_image(source, (output_num_pix, output_num_pix))

            # set exposures
            self.lens_exposure = lens
            self.source_exposure = source
        else:
            self.lens_exposure, self.source_exposure = None, None
