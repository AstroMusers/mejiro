import numpy as np
import time
import warnings

from mejiro.utils import util


class Exposure:

    def __init__(self, 
                 synthetic_image, 
                 exposure_time, 
                 engine='galsim', 
                 engine_params={}, 
                 psf=None, 
                 verbose=True
                 ):
        
        start = time.time()

        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time
        self.engine = engine
        self.verbose = verbose
        self.noise = None

        if engine == 'galsim':
            from mejiro.engines.galsim_engine import GalSimEngine

            self.noise = GalSimEngine.get_empty_image(self.synthetic_image.num_pix,
                                                       self.synthetic_image.pixel_scale)

            if self.synthetic_image.instrument_name == 'Roman':
                # get exposure
                results, self.psf, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = GalSimEngine.get_roman_exposure(
                    synthetic_image, exposure_time, psf, engine_params, self.verbose)

                # sum noise
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.reciprocity_failure is not None: self.noise += self.reciprocity_failure
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.nonlinearity is not None: self.noise += self.nonlinearity
                if self.ipc is not None: self.noise += self.ipc
                if self.read_noise is not None: self.noise += self.read_noise

                # it's confusing for all detector effects to be type galsim.Image and the noise attribute to be an ndarray, but for comparison across engines, the noise should be an array and the detector effects should be Images so they can be passed in as engine params
                self.noise = self.noise.array

            elif self.synthetic_image.instrument_name == 'HWO':
                # get exposure
                results, self.psf, self.sky_background, self.poisson_noise, self.dark_noise, self.read_noise = GalSimEngine.get_hwo_exposure(
                    synthetic_image, exposure_time, psf, engine_params, self.verbose)

                # sum noise
                if self.sky_background is not None: self.noise += self.sky_background
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.read_noise is not None: self.noise += self.read_noise
                
            else:
                self.instrument_not_available_error(engine)

        elif engine == 'lenstronomy':
            from mejiro.engines.lenstronomy_engine import LenstronomyEngine

            self.noise = np.zeros_like(self.synthetic_image.image)

            if self.synthetic_image.instrument_name == 'Roman':
                # get exposure
                results, self.psf, self.noise = LenstronomyEngine.get_roman_exposure(synthetic_image, exposure_time,
                                                                                      psf, engine_params, self.verbose)
            else:
                self.instrument_not_available_error(engine)

        elif engine == 'pandeia':
            from mejiro.engines.pandeia_engine import PandeiaEngine

            # warn that PSF isn't gonna do anything
            if psf is not None:
                warnings.warn('PSF is not used in the Pandeia engine')

            # get exposure
            results, self.psf, self.noise = PandeiaEngine.get_roman_exposure(synthetic_image, exposure_time, psf,
                                                                              engine_params, self.verbose)

            # TODO temporarily set noise to zeros until I can grab the noise that Pandeia generates
            self.noise = np.zeros((self.synthetic_image.num_pix, self.synthetic_image.num_pix))

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

        Exposure.crop_edge_effects(self.exposure, pad=3)  # crop off edge effects (e.g., IPC)
        if np.any(self.exposure < 0):
            raise ValueError('Negative pixel values in final image')

        if self.synthetic_image.pieces:
            Exposure.crop_edge_effects(self.lens_exposure, pad=3)
            Exposure.crop_edge_effects(self.source_exposure, pad=3)
            if np.any(self.lens_exposure < 0):
                raise ValueError('Negative pixel values in lens image')
            if np.any(self.source_exposure < 0):
                raise ValueError('Negative pixel values in source image')

        end = time.time()
        self.calc_time = end - start
        if self.verbose:
            print(f'Exposure calculation time with {self.engine} engine: {util.calculate_execution_time(start, end)}')

    def plot(self, savepath=None):
        import matplotlib.pyplot as plt

        plt.imshow(np.log10(self.exposure))
        plt.title(f'{self.synthetic_image.strong_lens.name}: {self.synthetic_image.instrument_name} {self.synthetic_image.band} band, {self.exposure_time} s exposure {self.exposure.shape}')
        cbar = plt.colorbar()
        cbar.set_label(r'log$_{10}$(Counts)')
        plt.xlabel('x [Pixels]')
        plt.ylabel('y [Pixels]')
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

    def instrument_not_available_error(self, engine):
        raise ValueError(
            f'Instrument "{self.synthetic_image.instrument_name}" not available for engine "{engine}."')

    @staticmethod
    def crop_edge_effects(image, pad):
        num_pix = image.shape[0]
        assert num_pix % 2 != 0, 'Image has even number of pixels'
        output_num_pix = num_pix - pad
        return util.center_crop_image(image, (output_num_pix, output_num_pix))
