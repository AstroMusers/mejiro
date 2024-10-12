import numpy as np
import time

from mejiro.utils import util


class Exposure:

    def __init__(self, synthetic_image, exposure_time, engine='galsim', engine_params=None, psf=None, verbose=True,
                 **kwargs):
        start = time.time()

        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time
        self.verbose = verbose

        if engine == 'galsim':
            from mejiro.engines import galsim_engine

            if self.synthetic_image.instrument.name == 'Roman':
                # validate engine params and set defaults
                if engine_params is None:
                    engine_params = galsim_engine.default_roman_engine_params()
                else:
                    engine_params = galsim_engine.validate_roman_engine_params(engine_params)

                results, self.psf, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = galsim_engine.get_roman_exposure(
                    synthetic_image, exposure_time, psf, engine_params, self.verbose, **kwargs)

            elif self.synthetic_image.instrument.name == 'HWO':
                # validate engine params and set defaults
                if engine_params is None:
                    engine_params = galsim_engine.default_hwo_engine_params()
                else:
                    engine_params = galsim_engine.validate_hwo_engine_params(engine_params)

                results, self.psf, self.poisson_noise, self.dark_noise, self.read_noise = galsim_engine.get_hwo_exposure(
                    synthetic_image, exposure_time, psf, engine_params, self.verbose, **kwargs)

        elif engine == 'pandeia':
            raise NotImplementedError('Pandeia engine not yet implemented')
            # TODO mejiro.engines import pandeia_engine
        elif engine == 'romanisim':
            raise NotImplementedError('romanisim engine not yet implemented')
            # TODO mejiro.engines import romanisim_engine
        else:
            raise ValueError(f'Engine "{engine}" not recognized')
        
        self.engine = engine
        self.engine_params = engine_params

        if self.synthetic_image.pieces:
            image, lens_image, source_image = results
        else:
            image = results
            self.lens_exposure, self.source_exposure = None, None

        exposure = image.array
        Exposure.crop_edge_effects(exposure)  # crop off edge effects (e.g., IPC)
        if np.any(exposure < 0):
            raise ValueError('Negative pixel values in final image')
        self.exposure = exposure

        if self.synthetic_image.pieces:
            lens_exposure = lens_image.array
            source_exposure = source_image.array
            Exposure.crop_edge_effects(lens_exposure)
            Exposure.crop_edge_effects(source_exposure)
            if np.any(lens_exposure < 0):
                raise ValueError('Negative pixel values in lens image')
            if np.any(source_exposure < 0):
                raise ValueError('Negative pixel values in source image')
            self.lens_exposure = lens_exposure
            self.source_exposure = source_exposure
        
        end = time.time()
        self.calc_time = end - start
        if self.verbose:
            print(f'Exposure calculation time: {util.calculate_execution_time(start, end)}')

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
