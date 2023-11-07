import copy

import lenstronomy.Plots.plot_util as plot_util
import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI


def simulate_image(kwargs_model, kwargs_numerics, kwargs_lens, kwargs_lens_light, kwargs_source_light):
    roman_f062 = Roman(band='F062', psf_type='GAUSSIAN', survey_mode='wide_area')
    # roman_z087 = Roman(band='F087', psf_type='GAUSSIAN', survey_mode='microlensing')
    roman_y106 = Roman(band='F106', psf_type='GAUSSIAN', survey_mode='wide_area')
    roman_j129 = Roman(band='F129', psf_type='GAUSSIAN', survey_mode='wide_area')
    roman_h158 = Roman(band='F158', psf_type='GAUSSIAN', survey_mode='wide_area')
    roman_f184 = Roman(band='F184', psf_type='GAUSSIAN', survey_mode='wide_area')
    # roman_w146 = Roman(band='F146', psf_type='GAUSSIAN', survey_mode='microlensing')

    kwargs_f062 = roman_f062.kwargs_single_band()
    kwargs_f106 = roman_y106.kwargs_single_band()
    kwargs_f129 = roman_j129.kwargs_single_band()
    kwargs_f158 = roman_h158.kwargs_single_band()
    kwargs_f184 = roman_f184.kwargs_single_band()

    roman_pixel_scale = roman_f062.kwargs_single_band().get('pixel_scale')
    roman_num_pix = int(5 / roman_pixel_scale)

    sim_f062 = SimAPI(numpix=roman_num_pix, kwargs_single_band=kwargs_f062, kwargs_model=kwargs_model)
    sim_f106 = SimAPI(numpix=roman_num_pix, kwargs_single_band=kwargs_f106, kwargs_model=kwargs_model)
    sim_f129 = SimAPI(numpix=roman_num_pix, kwargs_single_band=kwargs_f129, kwargs_model=kwargs_model)
    sim_f158 = SimAPI(numpix=roman_num_pix, kwargs_single_band=kwargs_f158, kwargs_model=kwargs_model)
    sim_f184 = SimAPI(numpix=roman_num_pix, kwargs_single_band=kwargs_f184, kwargs_model=kwargs_model)

    imsim_kwargs_f062 = sim_f062.image_model_class(kwargs_numerics)
    imsim_kwargs_f106 = sim_f106.image_model_class(kwargs_numerics)
    imsim_kwargs_f129 = sim_f129.image_model_class(kwargs_numerics)
    imsim_kwargs_f158 = sim_f158.image_model_class(kwargs_numerics)
    imsim_kwargs_f184 = sim_f184.image_model_class(kwargs_numerics)

    # set lens light magnitudes
    lens_light_mag_f062 = 14.
    lens_light_mag_f106 = 14.
    lens_light_mag_f129 = 15.
    lens_light_mag_f158 = 16.
    lens_light_mag_f184 = 16.

    # set source light magnitudes
    source_mag_f062 = 18.
    source_mag_f106 = 18.
    source_mag_f129 = 17.
    source_mag_f158 = 16.
    source_mag_f184 = 16.

    # lens light
    kwargs_lens_light_mag_f062 = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_f062[0]['amp']
    kwargs_lens_light_mag_f062[0]['magnitude'] = lens_light_mag_f062

    kwargs_lens_light_mag_f106 = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_f106[0]['amp']
    kwargs_lens_light_mag_f106[0]['magnitude'] = lens_light_mag_f106

    kwargs_lens_light_mag_f129 = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_f129[0]['amp']
    kwargs_lens_light_mag_f129[0]['magnitude'] = lens_light_mag_f129

    kwargs_lens_light_mag_f158 = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_f158[0]['amp']
    kwargs_lens_light_mag_f158[0]['magnitude'] = lens_light_mag_f158

    kwargs_lens_light_mag_f184 = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_f184[0]['amp']
    kwargs_lens_light_mag_f184[0]['magnitude'] = lens_light_mag_f184

    # source light
    kwargs_source_mag_f062 = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_f062[0]['amp']
    kwargs_source_mag_f062[0]['magnitude'] = source_mag_f062

    kwargs_source_mag_f106 = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_f106[0]['amp']
    kwargs_source_mag_f106[0]['magnitude'] = source_mag_f106

    kwargs_source_mag_f129 = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_f129[0]['amp']
    kwargs_source_mag_f129[0]['magnitude'] = source_mag_f129

    kwargs_source_mag_f158 = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_f158[0]['amp']
    kwargs_source_mag_f158[0]['magnitude'] = source_mag_f158

    kwargs_source_mag_f184 = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_f184[0]['amp']
    kwargs_source_mag_f184[0]['magnitude'] = source_mag_f184

    kwargs_lens_light_f062, kwargs_source_f062, _ = sim_f062.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_f062,
        kwargs_source_mag=kwargs_source_mag_f062)
    kwargs_lens_light_f106, kwargs_source_f106, _ = sim_f106.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_f106,
        kwargs_source_mag=kwargs_source_mag_f106)
    kwargs_lens_light_f129, kwargs_source_f129, _ = sim_f129.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_f129,
        kwargs_source_mag=kwargs_source_mag_f129)
    kwargs_lens_light_f158, kwargs_source_f158, _ = sim_f158.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_f158,
        kwargs_source_mag=kwargs_source_mag_f158)
    kwargs_lens_light_f184, kwargs_source_f184, _ = sim_f184.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_f184,
        kwargs_source_mag=kwargs_source_mag_f184)

    image_f062 = imsim_kwargs_f062.image(kwargs_lens=kwargs_lens,
                                         kwargs_lens_light=kwargs_lens_light_f062,
                                         kwargs_source=kwargs_source_f062)
    image_f106 = imsim_kwargs_f106.image(kwargs_lens=kwargs_lens,
                                         kwargs_lens_light=kwargs_lens_light_f106,
                                         kwargs_source=kwargs_source_f106)
    image_f129 = imsim_kwargs_f129.image(kwargs_lens=kwargs_lens,
                                         kwargs_lens_light=kwargs_lens_light_f129,
                                         kwargs_source=kwargs_source_f129)
    image_f158 = imsim_kwargs_f158.image(kwargs_lens=kwargs_lens,
                                         kwargs_lens_light=kwargs_lens_light_f158,
                                         kwargs_source=kwargs_source_f158)
    image_f184 = imsim_kwargs_f184.image(kwargs_lens=kwargs_lens,
                                         kwargs_lens_light=kwargs_lens_light_f184,
                                         kwargs_source=kwargs_source_f184)

    # add noise
    image_f062 += sim_f062.noise_for_model(model=image_f062)
    image_f106 += sim_f106.noise_for_model(model=image_f106)
    image_f129 += sim_f129.noise_for_model(model=image_f129)
    image_f158 += sim_f158.noise_for_model(model=image_f158)
    image_f184 += sim_f184.noise_for_model(model=image_f184)

    roman_image = np.zeros((image_f062.shape[0], image_f062.shape[1], 3), dtype=float)
    # roman_image[:,:,0] = plot_util.sqrt(image_f062, scale_min=0, scale_max=10000)
    roman_image[:, :, 0] = plot_util.sqrt(image_f106, scale_min=0, scale_max=10000)
    roman_image[:, :, 1] = plot_util.sqrt(image_f129, scale_min=0, scale_max=10000)
    roman_image[:, :, 2] = plot_util.sqrt(image_f158, scale_min=0, scale_max=10000)
    # roman_image[:,:,2] = plot_util.sqrt(image_f184, scale_min=0, scale_max=10000)

    return roman_image
