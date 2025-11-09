import numpy as np
from lenstronomy.SimulationAPI.sim_api import SimAPI

from mejiro.engines.engine import Engine
from mejiro.utils import util


class LenstronomyEngine(Engine):
    @staticmethod
    def defaults(engine_params, instrument_name):
        if 'kwargs_numerics' not in engine_params:
            engine_params |= {
                    'kwargs_numerics': {
                    'supersampling_factor': 5,
                    'compute_mode': 'regular'
                }
            }
        if 'noise' not in engine_params:
            engine_params |= {
                'noise': True
            }
        if 'obs_config_kwargs' not in engine_params:
            if instrument_name.lower() == 'roman':
                engine_params |= {
                    'obs_config_kwargs': {
                        'band': 'F129', 
                        'psf_type': 'PIXEL', 
                        'survey_mode': 'wide_area'
                    }
                }
            elif instrument_name.lower() == 'hst':
                engine_params |= {
                    'obs_config_kwargs': {
                        'band': 'WFC3_F160W', 
                        'psf_type': 'PIXEL', 
                        'coadd_years': None
                    }
                }
            elif instrument_name.lower() == 'lsst':
                engine_params |= {
                    'obs_config_kwargs': {
                        'band': 'i', 
                        'psf_type': 'GAUSSIAN', 
                        'coadd_years': 10
                    }
                }
            else:
                Engine.instrument_not_supported(instrument_name)

        return engine_params


    @staticmethod
    def validate_engine_params(engine_params):
        # TODO implement
        pass


    @staticmethod
    def get_exposure(synthetic_image, exposure_time, engine_params=None, verbose=False):
        engine_params = LenstronomyEngine.defaults(engine_params, synthetic_image.instrument_name)

        strong_lens = synthetic_image.strong_lens

        if synthetic_image.instrument_name == 'Roman':
            from lenstronomy.SimulationAPI.ObservationConfig import Roman
            obs_config = Roman.Roman(**engine_params['obs_config_kwargs'])
        elif synthetic_image.instrument_name == 'HST':
            from lenstronomy.SimulationAPI.ObservationConfig import HST
            obs_config = HST.HST(**engine_params['obs_config_kwargs'])
        elif synthetic_image.instrument_name == 'LSST':
            from lenstronomy.SimulationAPI.ObservationConfig import LSST
            obs_config = LSST.LSST(**engine_params['obs_config_kwargs'])
        # elif synthetic_image.instrument_name == 'Euclid':
        #     from lenstronomy.SimulationAPI.ObservationConfig import Euclid
        #     obs_config = Euclid.Euclid(**engine_params['obs_config_kwargs'])
        # elif synthetic_image.instrument_name == 'DES':
        #     from lenstronomy.SimulationAPI.ObservationConfig import DES
        #     obs_config = DES.DES(**engine_params['obs_config_kwargs'])
        # elif synthetic_image.instrument_name == 'JWST':
        #     from lenstronomy.SimulationAPI.ObservationConfig import JWST
        #     obs_config = JWST.JWST(**engine_params['obs_config_kwargs'])
        else:
            Engine.instrument_not_supported(synthetic_image.instrument_name)
        # obs_config.obs['num_exposures'] = 1  # set number of exposures to 1 cf. 96

        obs_config.obs['exposure_time'] = exposure_time

        sim_api = SimAPI(numpix=synthetic_image.num_pix,
                        kwargs_single_band=obs_config.kwargs_single_band(),
                        kwargs_model=strong_lens.kwargs_model)

        imsim = sim_api.image_model_class(engine_params['kwargs_numerics'])

        total_image = imsim.image(kwargs_lens=strong_lens.kwargs_lens, 
                                  kwargs_source=strong_lens.kwargs_source, 
                                  kwargs_lens_light=strong_lens.kwargs_lens_light,
                                  kwargs_ps=strong_lens.kwargs_ps,
                                  kwargs_extinction=strong_lens.kwargs_extinction,
                                  kwargs_special=strong_lens.kwargs_special,
                                  unconvolved=False,
                                  source_add=True,
                                  lens_light_add=True,
                                  point_source_add=True)

        if isinstance(engine_params['noise'], bool) and engine_params['noise']:
            noise = sim_api.noise_for_model(model=total_image)
            total_image += noise
        elif isinstance(engine_params['noise'], (list, tuple, np.ndarray)):
            noise = engine_params['noise']
            total_image += noise
        else:
            noise = None

        # if any unphysical negative pixels exist, set them to minimum value
        total_image = util.replace_negatives(total_image, util.smallest_non_negative_element(total_image))

        if synthetic_image.pieces:
            lens_surface_brightness = imsim.image(
                kwargs_lens=strong_lens.kwargs_lens, 
                kwargs_source=strong_lens.kwargs_source, 
                kwargs_lens_light=strong_lens.kwargs_lens_light,
                kwargs_ps=strong_lens.kwargs_ps,
                kwargs_extinction=strong_lens.kwargs_extinction,
                kwargs_special=strong_lens.kwargs_special,
                source_add=False)
            source_surface_brightness = imsim.image(
                kwargs_lens=strong_lens.kwargs_lens,
                kwargs_source=strong_lens.kwargs_source,
                kwargs_lens_light=strong_lens.kwargs_lens_light,
                kwargs_ps=strong_lens.kwargs_ps,
                kwargs_extinction=strong_lens.kwargs_extinction,
                kwargs_special=strong_lens.kwargs_special,
                source_add=True,
                lens_light_add=False)
            lens_surface_brightness = util.replace_negatives_with_zeros(lens_surface_brightness)
            source_surface_brightness = util.replace_negatives_with_zeros(source_surface_brightness)
            return (total_image, lens_surface_brightness, source_surface_brightness), noise
        else:
            return total_image, noise
