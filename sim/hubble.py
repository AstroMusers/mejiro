import copy

import lenstronomy.Plots.plot_util as plot_util
import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.sim_api import SimAPI


def simulate_image(kwargs_model, kwargs_numerics, kwargs_lens, kwargs_lens_light, kwargs_source_light):
    hst_wfc3_f160w = HST(band='WFC3_F160W', psf_type='GAUSSIAN', coadd_years=None)

    kwargs_wfc3_f160w = hst_wfc3_f160w.kwargs_single_band()

    hst_pixel_scale = hst_wfc3_f160w.kwargs_single_band().get('pixel_scale')
    hst_num_pix = int(5 / hst_pixel_scale)

    sim_wfc3_f160w = SimAPI(numpix=hst_num_pix, kwargs_single_band=kwargs_wfc3_f160w, kwargs_model=kwargs_model)

    imsim_kwargs_wfc3_f160w = sim_wfc3_f160w.image_model_class(kwargs_numerics)

    # set magnitudes
    lens_light_mag_wfc3_f160w = 14.
    source_mag_wfc3_f160w = 19.

    # lens light
    kwargs_lens_light_mag_wfc3_f160w = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_wfc3_f160w[0]['amp']
    kwargs_lens_light_mag_wfc3_f160w[0]['magnitude'] = lens_light_mag_wfc3_f160w

    # source light
    kwargs_source_mag_wfc3_f160w = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_wfc3_f160w[0]['amp']
    kwargs_source_mag_wfc3_f160w[0]['magnitude'] = source_mag_wfc3_f160w

    kwargs_lens_light_wfc3_f160w, kwargs_source_wfc3_f160w, _ = sim_wfc3_f160w.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_wfc3_f160w,
        kwargs_source_mag=kwargs_source_mag_wfc3_f160w)

    image_wfc3_f160w = imsim_kwargs_wfc3_f160w.image(kwargs_lens=kwargs_lens,
                                                     kwargs_lens_light=kwargs_lens_light_wfc3_f160w,
                                                     kwargs_source=kwargs_source_wfc3_f160w)

    # add noise
    image_wfc3_f160w += sim_wfc3_f160w.noise_for_model(model=image_wfc3_f160w)

    hst_image = np.zeros((image_wfc3_f160w.shape[0], image_wfc3_f160w.shape[1], 1), dtype=float)
    hst_image[:, :, 0] = plot_util.sqrt(image_wfc3_f160w, scale_min=0, scale_max=10000)

    return hst_image
