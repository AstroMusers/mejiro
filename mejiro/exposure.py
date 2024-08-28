import numpy as np
import galsim

from mejiro.helpers import gs, psf
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
        self.lens_flux_cps = np.sum(self.synthetic_image.strong_lens.lens_light_model_class.total_flux(
            [self.synthetic_image.strong_lens.kwargs_lens_light_amp_dict[self.synthetic_image.band]]))
        self.source_flux_cps = np.sum(self.synthetic_image.strong_lens.source_model_class.total_flux(
            [self.synthetic_image.strong_lens.kwargs_source_amp_dict[self.synthetic_image.band]]))
        self.total_flux_cps = self.lens_flux_cps + self.source_flux_cps

        # create interpolated image
        self.interp = galsim.InterpolatedImage(galsim.Image(self.synthetic_image.image, xmin=0, ymin=0),
                                   scale=self.synthetic_image.pixel_scale,
                                   flux=self.total_flux_cps * self.exposure_time)

        tuple = self.synthetic_image.instrument.get_exposure(self.synthetic_image, self.interp, self.rng, self.exposure_time, **kwargs)
        if 'return_noise' in kwargs and kwargs['return_noise']:
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
        self.exposure = final
