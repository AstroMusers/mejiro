from copy import deepcopy
import numpy as np
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Plots import plot_util


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
    kwargs_lens_light_mag_wfc3_f160w = [deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_wfc3_f160w[0]['amp']
    kwargs_lens_light_mag_wfc3_f160w[0]['magnitude'] = lens_light_mag_wfc3_f160w

    # source light
    kwargs_source_mag_wfc3_f160w = [deepcopy(kwargs_source_light)]
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


def get_roman_sim(lens, noise=True, side=5.):
    kwargs_numerics = {'point_source_supersampling_factor': 1}

    roman_g = Roman(band='F062', psf_type='PIXEL', survey_mode='wide_area')
    roman_r = Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')
    roman_i = Roman(band='F184', psf_type='PIXEL', survey_mode='wide_area')

    kwargs_b_band = roman_g.kwargs_single_band()
    kwargs_g_band = roman_r.kwargs_single_band()
    kwargs_r_band = roman_i.kwargs_single_band()

    # set number of pixels from pixel scale
    pixel_scale = kwargs_g_band['pixel_scale']
    numpix = int(round(side / pixel_scale))

    sim_b = SimAPI(numpix=numpix, kwargs_single_band=kwargs_b_band, kwargs_model=lens.kwargs_model)
    sim_g = SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=lens.kwargs_model)
    sim_r = SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=lens.kwargs_model)

    imsim_b = sim_b.image_model_class(kwargs_numerics)
    imsim_g = sim_g.image_model_class(kwargs_numerics)
    imsim_r = sim_r.image_model_class(kwargs_numerics)

    # TODO TEMP for now, just set flat spectrum i.e. same brightness across filters
    kwargs_lens_light_mag_g = lens.kwargs_lens_light
    kwargs_lens_light_mag_r = lens.kwargs_lens_light
    kwargs_lens_light_mag_i = lens.kwargs_lens_light
    kwargs_source_mag_g = lens.kwargs_source
    kwargs_source_mag_r = lens.kwargs_source
    kwargs_source_mag_i = lens.kwargs_source

    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g, _ = sim_b.magnitude2amplitude(kwargs_lens_light_mag_g,
                                                                        kwargs_source_mag_g, None)
    kwargs_lens_light_r, kwargs_source_r, _ = sim_g.magnitude2amplitude(kwargs_lens_light_mag_r,
                                                                        kwargs_source_mag_r, None)
    kwargs_lens_light_i, kwargs_source_i, _ = sim_r.magnitude2amplitude(kwargs_lens_light_mag_i,
                                                                        kwargs_source_mag_i, None)

    # convert from physical to lensing units
    kwargs_lens_lensing_units = sim_b.physical2lensing_conversion(kwargs_mass=lens.kwargs_lens)

    # create images in each filter
    image_b = imsim_b.image(kwargs_lens_lensing_units, kwargs_source_g, kwargs_lens_light_g)
    image_g = imsim_g.image(kwargs_lens_lensing_units, kwargs_source_r, kwargs_lens_light_r)
    image_r = imsim_r.image(kwargs_lens_lensing_units, kwargs_source_i, kwargs_lens_light_i)

    # add noise
    if noise:
        image_b += sim_b.noise_for_model(model=image_b)
        image_g += sim_g.noise_for_model(model=image_g)
        image_r += sim_r.noise_for_model(model=image_r)

    # pick an image to return as the single filter image
    image = image_g

    # assemble the color image
    rgb_image = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)

    # scale_max=10000
    def _scale_max(image):
        flat = image.flatten()
        flat.sort()
        scale_max = flat[int(len(flat) * 0.95)]
        return scale_max

    rgb_image[:, :, 0] = plot_util.sqrt(image_b, scale_min=0, scale_max=_scale_max(image_b))
    rgb_image[:, :, 1] = plot_util.sqrt(image_g, scale_min=0, scale_max=_scale_max(image_g))
    rgb_image[:, :, 2] = plot_util.sqrt(image_r, scale_min=0, scale_max=_scale_max(image_r))

    return image, rgb_image, sim_b.data_class


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
    kwargs_lens_light_mag_g = [deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_g[0]['amp']
    kwargs_lens_light_mag_g[0]['magnitude'] = lens_light_mag_g

    kwargs_lens_light_mag_r = [deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_r[0]['amp']
    kwargs_lens_light_mag_r[0]['magnitude'] = lens_light_mag_r

    kwargs_lens_light_mag_i = [deepcopy(kwargs_lens_light)]
    del kwargs_lens_light_mag_i[0]['amp']
    kwargs_lens_light_mag_i[0]['magnitude'] = lens_light_mag_i

    # source light
    kwargs_source_mag_g = [deepcopy(kwargs_source_light)]
    del kwargs_source_mag_g[0]['amp']
    kwargs_source_mag_g[0]['magnitude'] = source_mag_g

    kwargs_source_mag_r = [deepcopy(kwargs_source_light)]
    del kwargs_source_mag_r[0]['amp']
    kwargs_source_mag_r[0]['magnitude'] = source_mag_r

    kwargs_source_mag_i = [deepcopy(kwargs_source_light)]
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
