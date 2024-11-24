import numpy as np
import time

from mejiro.utils import util


class Exposure:

    def __init__(self, synthetic_image, exposure_time, engine='galsim', engine_params=None, psf=None, verbose=True,
                 **kwargs):
        start = time.time()

        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time
        self.engine = engine
        self.verbose = verbose
        self.noise = None

        if engine == 'galsim':
            from mejiro.engines import galsim_engine

            self.noise = galsim_engine.get_empty_image(self.synthetic_image.native_num_pix,
                                                       self.synthetic_image.native_pixel_scale)

            if self.synthetic_image.instrument.name == 'Roman':
                # validate engine params and set defaults
                if engine_params is None:
                    engine_params = galsim_engine.default_roman_engine_params()
                else:
                    engine_params = galsim_engine.validate_roman_engine_params(engine_params)

                # get exposure
                results, self.psf, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = galsim_engine.get_roman_exposure(
                    synthetic_image, exposure_time, psf, engine_params, self.verbose, **kwargs)

                # sum noise
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.reciprocity_failure is not None: self.noise += self.reciprocity_failure
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.nonlinearity is not None: self.noise += self.nonlinearity
                if self.ipc is not None: self.noise += self.ipc
                if self.read_noise is not None: self.noise += self.read_noise

                # TODO it's confusing for all detector effects to be type galsim.Image and the noise attribute to be an ndarray, but for comparison across engines the noise should be an array and the detector effects should be Images so they can be passed in as engine params
                self.noise = self.noise.array
            elif self.synthetic_image.instrument.name == 'HWO':
                # validate engine params and set defaults
                if engine_params is None:
                    engine_params = galsim_engine.default_hwo_engine_params()
                else:
                    engine_params = galsim_engine.validate_hwo_engine_params(engine_params)

                # get exposure
                results, self.psf, self.poisson_noise, self.dark_noise, self.read_noise = galsim_engine.get_hwo_exposure(
                    synthetic_image, exposure_time, psf, engine_params, self.verbose, **kwargs)

                # sum noise
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.read_noise is not None: self.noise += self.read_noise
            else:
                self.instrument_not_available_error(engine)

        elif engine == 'lenstronomy':
            from mejiro.engines import lenstronomy_engine

            self.noise = np.zeros_like(self.synthetic_image.image)

            if self.synthetic_image.instrument.name == 'Roman':
                # validate engine params and set defaults
                if engine_params is None:
                    engine_params = lenstronomy_engine.default_roman_engine_params()
                else:
                    engine_params = lenstronomy_engine.validate_roman_engine_params(engine_params)

                # get exposure
                results, self.psf, self.noise = lenstronomy_engine.get_roman_exposure(synthetic_image, exposure_time,
                                                                                      psf, engine_params, self.verbose,
                                                                                      **kwargs)
            else:
                self.instrument_not_available_error(engine)

        elif engine == 'pandeia':
            # raise NotImplementedError('Pandeia engine not yet implemented')
            from mejiro.engines import pandeia_engine

            # TODO warn that PSF isn't gonna do anything

            # validate engine params and set defaults
            if engine_params is None:
                engine_params = pandeia_engine.default_roman_engine_params()
            else:
                engine_params = pandeia_engine.validate_roman_engine_params(engine_params)

            # get exposure
            results, self.psf, self.noise = pandeia_engine.get_roman_exposure(synthetic_image, exposure_time, psf,
                                                                              engine_params, self.verbose, **kwargs)

            # TODO temporarily set noise to zeros until pandeia noise is implemented
            self.noise = np.zeros_like(self.synthetic_image.image)

        elif engine == 'romanisim':
            raise NotImplementedError('romanisim engine not yet implemented')
            # TODO from mejiro.engines import romanisim_engine

        else:
            raise ValueError(f'Engine "{engine}" not recognized')

        # once engine params have been defaulted and validated, set them as an attribute
        self.engine_params = engine_params

        # set image and expoure attributes
        if self.engine == 'galsim':
            if self.synthetic_image.pieces:
                self.image, self.lens_image, self.source_image = results
                self.exposure, self.lens_exposure, self.source_exposure = self.image.array, self.lens_image.array, self.source_image.array
            else:
                self.image, self.lens_image, self.source_image = results, None, None
                self.exposure, self.lens_exposure, self.source_exposure = self.image.array, None, None
        else:
            if self.synthetic_image.pieces:
                self.exposure, self.lens_exposure, self.source_exposure = results
            else:
                self.exposure, self.lens_exposure, self.source_exposure = results, None, None

        Exposure.crop_edge_effects(self.exposure)  # crop off edge effects (e.g., IPC)
        if np.any(self.exposure < 0):
            raise ValueError('Negative pixel values in final image')

        if self.synthetic_image.pieces:
            Exposure.crop_edge_effects(self.lens_exposure)
            Exposure.crop_edge_effects(self.source_exposure)
            if np.any(self.lens_exposure < 0):
                raise ValueError('Negative pixel values in lens image')
            if np.any(self.source_exposure < 0):
                raise ValueError('Negative pixel values in source image')

        end = time.time()
        self.calc_time = end - start
        if self.verbose:
            print(f'Exposure calculation time with {self.engine} engine: {util.calculate_execution_time(start, end)}')

    def instrument_not_available_error(self, engine):
        raise ValueError(
            f'Instrument "{self.synthetic_image.instrument.name}" not available for engine "{engine}." Available engines are {self.synthetic_image.instrument.engines}')

    @staticmethod
    def crop_edge_effects(image):
        num_pix = image.shape[0]
        assert num_pix % 2 != 0, 'Image has even number of pixels'
        output_num_pix = num_pix - 3
        return util.center_crop_image(image, (output_num_pix, output_num_pix))

    @property
    def get_exposure_time(self):
        return self.exposure_time

    @property
    def get_exposure(self):
        return self.exposure

    @property
    def get_lens_exposure(self):
        return self.lens_exposure

    @property
    def get_source_exposure(self):
        return self.source_exposure
