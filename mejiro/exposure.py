import logging
import numpy as np
import time
import warnings

from mejiro.utils import util
from mejiro.analysis.snr_calculation import get_snr

logger = logging.getLogger(__name__)


class Exposure:

    def __init__(self, 
                 synthetic_image, 
                 exposure_time, 
                 engine='galsim',
                 engine_params={}
                 ):
        
        start = time.time()

        self.synthetic_image = synthetic_image
        self.exposure_time = exposure_time
        self.engine = engine
        self.noise = None

        if engine == 'galsim':
            from mejiro.engines.galsim_engine import GalSimEngine

            self.noise = GalSimEngine.get_empty_image(self.synthetic_image.num_pix,
                                                       self.synthetic_image.pixel_scale)

            if self.synthetic_image.instrument_name == 'Roman':
                # get exposure
                results, self.sky_background, self.poisson_noise, self.reciprocity_failure, self.dark_noise, self.nonlinearity, self.ipc, self.read_noise = GalSimEngine.get_roman_exposure(
                    synthetic_image, exposure_time, engine_params)

                # sum noise
                if self.sky_background is not None: self.noise += self.sky_background
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.reciprocity_failure is not None: self.noise += self.reciprocity_failure
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.nonlinearity is not None: self.noise += self.nonlinearity
                if self.ipc is not None: self.noise += self.ipc
                if self.read_noise is not None: self.noise += self.read_noise

            elif self.synthetic_image.instrument_name == 'HWO' or self.synthetic_image.instrument_name == 'JWST' or self.synthetic_image.instrument_name == 'HST':
                # get exposure
                results, self.sky_background, self.poisson_noise, self.dark_noise, self.read_noise = GalSimEngine.get_exposure(
                    synthetic_image, exposure_time, engine_params)
                
                # sum noise
                if self.sky_background is not None: self.noise += self.sky_background
                if self.poisson_noise is not None: self.noise += self.poisson_noise
                if self.dark_noise is not None: self.noise += self.dark_noise
                if self.read_noise is not None: self.noise += self.read_noise
                
            else:
                self.instrument_not_available_error(engine)

            # write the noise out to a numpy array
            self.noise = self.noise.array  # it's confusing for all detector effects to be type galsim.Image and the noise attribute to be an ndarray, but for comparison across engines, the noise should be an array and the detector effects should be Images so they can be passed in as engine params

        elif engine == 'lenstronomy':
            raise NotImplementedError('Lenstronomy engine not yet implemented')

            from mejiro.engines.lenstronomy_engine import LenstronomyEngine

            self.noise = np.zeros_like(self.synthetic_image.data)

            # get exposure
            results, self.noise = LenstronomyEngine.get_exposure(
                synthetic_image=synthetic_image,
                exposure_time=exposure_time,
                engine_params=engine_params)
            # TODO conditional for supported instruments

        elif engine == 'pandeia':
            raise NotImplementedError('Pandeia engine not yet implemented')
        
            from mejiro.engines.pandeia_engine import PandeiaEngine

            # get exposure
            results, self.noise = PandeiaEngine.get_roman_exposure(synthetic_image, exposure_time, engine_params)

            # TODO temporarily set noise to zeros until I can grab the noise that Pandeia generates
            self.noise = np.zeros((self.synthetic_image.num_pix, self.synthetic_image.num_pix))

        elif engine == 'romanisim':
            raise NotImplementedError('romanisim engine not yet implemented')
            
            from mejiro.engines.romanisim_engine import RomanISimEngine

            if self.synthetic_image.instrument_name == 'Roman':
                results, self.noise = RomanISimEngine.get_roman_exposure(synthetic_image, exposure_time, engine_params)
                
            else:
                self.instrument_not_available_error(engine)

        else:
            raise ValueError(f'Engine "{engine}" not recognized')

        # once engine params have been defaulted and validated, set them as an attribute
        self.engine_params = engine_params

        # set image and expoure attributes
        if self.engine == 'galsim':
            if self.synthetic_image.pieces:
                self.image, self.lens_image, self.source_image = results
                self.data, self.lens_data, self.source_data = self.image.array, self.lens_image.array, self.source_image.array
            else:
                self.image, self.lens_image, self.source_image = results, None, None
                self.data, self.lens_data, self.source_data = self.image.array, None, None
        else:
            if self.synthetic_image.pieces:
                self.data, self.lens_data, self.source_data = results
            else:
                self.data, self.lens_data, self.source_data = results, None, None

        # crop off edge effects (e.g., IPC)
        Exposure.crop_edge_effects(self.data, pad=3)

        # check for negative pixels
        if np.any(self.data < 0):
            warnings.warn(f'Negative pixel values in final image. Setting {np.sum(self.data < 0)} pixels to 0')
            self.data[self.data < 0] = 0

        if self.synthetic_image.pieces:
            Exposure.crop_edge_effects(self.lens_data, pad=3)
            Exposure.crop_edge_effects(self.source_data, pad=3)
            if np.any(self.lens_data < 0):
                warnings.warn(f'Negative pixel values in lens image. Setting {np.sum(self.lens_data < 0)} pixels to 0')
                self.lens_data[self.lens_data < 0] = 0
            if np.any(self.source_data < 0):
                warnings.warn(f'Negative pixel values in source image. Setting {np.sum(self.source_data < 0)} pixels to 0')
                self.source_data[self.source_data < 0] = 0

        end = time.time()
        self.calc_time = end - start
        logger.info(f'Exposure calculation time with {self.engine} engine: {util.calculate_execution_time(start, end, unit="s")}')

    def __getstate__(self):
        state = self.__dict__.copy()
        # drop the SyntheticImage reference on pickle: it's already written to disk in step 04,
        # so embedding it here fully duplicates that output (~2.77 MB per Exposure).
        # step 06 loads the SyntheticImage from step 04's pickle directly.
        state['synthetic_image'] = None
        return state

    def get_snr(self, snr_per_pixel_threshold=1):
        return get_snr(self, snr_per_pixel_threshold=snr_per_pixel_threshold)[0]

    def plot(self, show_snr=False, savepath=None):
        import matplotlib.pyplot as plt

        plt.imshow(np.log10(self.data), origin='lower')

        title = f'{self.synthetic_image.strong_lens.name} (' + r'$z_{l}=$' + f'{self.synthetic_image.strong_lens.z_lens:.2f}, ' + r'$z_{s}=$' + f'{self.synthetic_image.strong_lens.z_source:.2f}' + f')\n{self.synthetic_image.instrument_name} {self.synthetic_image.band}, {self.exposure_time} s'
        if show_snr:
            snr = self.get_snr()
            title += f'\nSNR: {snr:.2f}'
        plt.title(title)
        cbar = plt.colorbar()
        cbar.set_label(r'log$_{10}$(Counts)')
        plt.xlabel('x [Pixels]')
        plt.ylabel('y [Pixels]')
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

    # def detailed_plot(self, savepath=None):
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #     axs[0].imshow(np.log10(self.data), cmap='viridis')
    #     axs[0].set_title(f'Exposure: {self.synthetic_image.instrument_name} {self.synthetic_image.band} band, {self.exposure_time} s')
    #     axs[0].set_xlabel('x [Pixels]')
    #     axs[0].set_ylabel('y [Pixels]')
    #     cbar = fig.colorbar(axs[0].images[0], ax=axs[0])
    #     cbar.set_label(r'log$_{10}$(Counts)')

    #     if self.lens_data is not None:
    #         axs[1].imshow(np.log10(self.lens_data), cmap='viridis')
    #         axs[1].set_title('Lens Image')
    #         axs[1].set_xlabel('x [Pixels]')
    #         axs[1].set_ylabel('y [Pixels]')
    #         cbar = fig.colorbar(axs[1].images[0], ax=axs[1])
    #         cbar.set_label(r'log$_{10}$(Counts)')

    #     if self.source_data is not None:
    #         axs[2].imshow(np.log10(self.source_data), cmap='viridis')
    #         axs[2].set_title('Source Image')
    #         axs[2].set_xlabel('x [Pixels]')
    #         axs[2].set_ylabel('y [Pixels]')
    #         cbar = fig.colorbar(axs[2].images[0], ax=axs[2])
    #         cbar.set_label(r'log$_{10}$(Counts)')

    #     plt.tight_layout()
    #     if savepath is not None:
    #         plt.savefig(savepath)
    #     plt.show()

    def instrument_not_available_error(self, engine):
        raise ValueError(
            f'Instrument "{self.synthetic_image.instrument_name}" not available for engine "{engine}."')

    @staticmethod
    def crop_edge_effects(image, pad):
        num_pix = image.shape[0]
        assert num_pix % 2 != 0, 'Image has even number of pixels'
        output_num_pix = num_pix - pad
        return util.center_crop_image(image, (output_num_pix, output_num_pix))
