import galsim
import numpy as np

from mejiro.utils import util


class Exposure:

    def __init__(self, synthetic_image, exposure_time, engine='galsim', engine_params=None, psf=None, verbose=True, **kwargs):
        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time
        self.verbose = verbose

        if engine == 'galsim':
            from mejiro.engines import galsim_engine

            if self.synthetic_image.instrument.name == 'Roman':
                self.image, self.psf, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = galsim_engine.get_roman_exposure(synthetic_image, exposure_time, psf, engine_params, self.verbose, **kwargs)

            elif self.synthetic_image.instrument.name == 'HWO':
                self.image, self.psf, self.poisson_noise, self.dark_noise, self.read_noise = galsim_engine.get_hwo_exposure(synthetic_image, exposure_time, psf, engine_params, self.verbose, **kwargs)

        elif engine == 'pandeia':
            raise NotImplementedError('Pandeia engine not yet implemented')
            # TODO mejiro.engines import pandeia_engine
        elif engine == 'romanisim':
            raise NotImplementedError('romanisim engine not yet implemented')
            # TODO mejiro.engines import romanisim_engine
        else:
            raise ValueError(f'Engine "{engine}" not recognized')

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

    @property
    def get_exposure_time(self):
        return self.exposure_time
    
    @property
    def get_exposure(self):
        return self.exposure
    