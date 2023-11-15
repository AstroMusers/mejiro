import copy

import lenstronomy.Plots.plot_util as plot_util
import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.sim_api import SimAPI


def simulate_image(kwargs_model, kwargs_numerics, kwargs_lens, kwargs_lens_light, kwargs_source_light):
    lsst_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
    lsst_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
    lsst_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)

    kwargs_g_band = lsst_g.kwargs_single_band()
    kwargs_r_band = lsst_r.kwargs_single_band()
    kwargs_i_band = lsst_i.kwargs_single_band()

    lsst_pixel_scale = lsst_g.kwargs_single_band().get('pixel_scale')
    lsst_num_pix = int(5 / lsst_pixel_scale)

    sim_g = SimAPI(numpix=lsst_num_pix, kwargs_single_band=kwargs_g_band, kwargs_model=kwargs_model)
    sim_r = SimAPI(numpix=lsst_num_pix, kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model)
    sim_i = SimAPI(numpix=lsst_num_pix, kwargs_single_band=kwargs_i_band, kwargs_model=kwargs_model)

    imsim_g = sim_g.image_model_class(kwargs_numerics)
    imsim_r = sim_r.image_model_class(kwargs_numerics)
    imsim_i = sim_i.image_model_class(kwargs_numerics)

    # g-band
    lens_light_mag_g = 14.
    source_mag_g = 19.

    # r-band
    lens_light_mag_r = 15.
    source_mag_r = 18.

    # i-band
    lens_light_mag_i = 16.
    source_mag_i = 17.

    # lens light
    kwargs_lens_light_mag_g = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_g[0]['amp']
    kwargs_lens_light_mag_g[0]['magnitude'] = lens_light_mag_g

    kwargs_lens_light_mag_r = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_r[0]['amp']
    kwargs_lens_light_mag_r[0]['magnitude'] = lens_light_mag_r

    kwargs_lens_light_mag_i = [copy.deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_i[0]['amp']
    kwargs_lens_light_mag_i[0]['magnitude'] = lens_light_mag_i

    # source light
    kwargs_source_mag_g = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_g[0]['amp']
    kwargs_source_mag_g[0]['magnitude'] = source_mag_g

    kwargs_source_mag_r = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_r[0]['amp']
    kwargs_source_mag_r[0]['magnitude'] = source_mag_r

    kwargs_source_mag_i = [copy.deepcopy(kwargs_source_light)]
    del kwargs_source_mag_i[0]['amp']
    kwargs_source_mag_i[0]['magnitude'] = source_mag_i

    kwargs_lens_light_g, kwargs_source_g, _ = sim_g.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_g,
        kwargs_source_mag=kwargs_source_mag_g)
    kwargs_lens_light_r, kwargs_source_r, _ = sim_r.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_r,
        kwargs_source_mag=kwargs_source_mag_r)
    kwargs_lens_light_i, kwargs_source_i, _ = sim_i.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag_i,
        kwargs_source_mag=kwargs_source_mag_i)

    image_g = imsim_g.image(kwargs_lens=kwargs_lens,
                            kwargs_lens_light=kwargs_lens_light_g,
                            kwargs_source=kwargs_source_g)
    image_r = imsim_r.image(kwargs_lens=kwargs_lens,
                            kwargs_lens_light=kwargs_lens_light_r,
                            kwargs_source=kwargs_source_r)
    image_i = imsim_i.image(kwargs_lens=kwargs_lens,
                            kwargs_lens_light=kwargs_lens_light_i,
                            kwargs_source=kwargs_source_i)

    # add noise
    image_g += sim_g.noise_for_model(model=image_g)
    image_r += sim_r.noise_for_model(model=image_r)
    image_i += sim_i.noise_for_model(model=image_i)

    lsst_image = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
    lsst_image[:, :, 0] = plot_util.sqrt(image_g, scale_min=0, scale_max=10000)
    lsst_image[:, :, 1] = plot_util.sqrt(image_r, scale_min=0, scale_max=10000)
    lsst_image[:, :, 2] = plot_util.sqrt(image_i, scale_min=0, scale_max=10000)

    return lsst_image
