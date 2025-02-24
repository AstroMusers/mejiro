from lenstronomy.SimulationAPI.ObservationConfig import HST, LSST, Roman, DES, Euclid
from lenstronomy.SimulationAPI.sim_api import SimAPI

from mejiro.engines.engine import Engine
from mejiro.utils import util


class LenstronomyEngine(Engine):
    @staticmethod
    def defaults(instrument_name):
        if instrument_name.casefold() == 'Roman':
            return {
                'kwargs_numerics': {
                    'supersampling_factor': 3,
                    'compute_mode': 'regular'
                },
                'noise': True
            }
        else:
            Engine().instrument_not_supported(instrument_name)


    @staticmethod
    def validate_engine_params(engine_params):
        # TODO implement
        pass


    @staticmethod
    def get_roman_exposure(synthetic_image, exposure_time, psf=None, engine_params=defaults('Roman'),
                        verbose=False, **kwargs):
        strong_lens = synthetic_image.strong_lens
        band = synthetic_image.band

        roman_obs_config = Roman.Roman(band=band,
                                    psf_type='PIXEL',
                                    survey_mode='wide_area')
        roman_obs_config.obs['num_exposures'] = 1  # set number of exposures to 1 cf. 96
        roman_obs_config.obs['exposure_time'] = exposure_time

        if psf is None:
            psf = roman_obs_config.obs['kernel_point_source']
        else:
            roman_obs_config.obs['kernel_point_source'] = psf

        sim_api = SimAPI(numpix=synthetic_image.native_num_pix,
                        kwargs_single_band=roman_obs_config.kwargs_single_band(),
                        kwargs_model=strong_lens.kwargs_model)

        imsim = sim_api.image_model_class(engine_params['kwargs_numerics'])

        kwargs_lens = strong_lens.kwargs_lens
        kwargs_lens_light = [strong_lens.kwargs_lens_light_amp_dict[band]]
        kwargs_source = [strong_lens.kwargs_source_amp_dict[band]]

        total_image = imsim.image(kwargs_lens, kwargs_source, kwargs_lens_light, None)

        if engine_params['noise']:
            noise = sim_api.noise_for_model(model=total_image)
            total_image += noise

        # if any unphysical negative pixels exist, set them to minimum value
        total_image = util.replace_negatives(total_image, util.smallest_non_negative_element(total_image))

        if synthetic_image.pieces:
            lens_surface_brightness = imsim.image(kwargs_lens,
                                                kwargs_source,
                                                kwargs_lens_light,
                                                None,
                                                source_add=False)
            source_surface_brightness = imsim.image(kwargs_lens,
                                                    kwargs_source,
                                                    kwargs_lens_light,
                                                    None,
                                                    lens_light_add=False)
            lens_surface_brightness = util.replace_negatives_with_zeros(lens_surface_brightness)
            source_surface_brightness = util.replace_negatives_with_zeros(source_surface_brightness)
            return total_image, lens_surface_brightness, source_surface_brightness, psf, noise
        else:
            return total_image, psf, noise
