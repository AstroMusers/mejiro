from copy import deepcopy

import numpy as np
from lenstronomy.Plots import plot_util
from lenstronomy.SimulationAPI.ObservationConfig import HST, LSST, Roman, DES, Euclid
from lenstronomy.SimulationAPI.sim_api import SimAPI


def get_image(lens, telescope, side):
    kwargs_lens_list, kwargs_source_list = set_up_magnitudes(lens)

    if telescope.lower() == 'roman':
        config_list = roman_config_list()
    elif telescope.lower() == 'roman_rgb':
        config_list = roman_rgb_config_list()
    elif telescope.lower() == 'hst':
        config_list = hst_config_list()
    elif telescope.lower() == 'lsst':
        config_list = lsst_config_list()
    elif telescope.lower() == 'des':
        config_list = des_config_list()
    elif telescope.lower() == 'euclid':
        config_list = euclid_config_list()
    else:
        raise Exception('Unsupported telescope')
    
    return simulate_rgb(lens, kwargs_lens_list, kwargs_source_list, config_list, side)


def set_up_magnitudes(lens):
    # r-band
    kwargs_lens_light_mag_r = lens.kwargs_lens_light
    kwargs_source_mag_r = lens.kwargs_source

    # g-band
    kwargs_lens_light_mag_g = deepcopy(kwargs_lens_light_mag_r)
    kwargs_lens_light_mag_g[0]['magnitude'] -= 1

    kwargs_source_mag_g = deepcopy(kwargs_source_mag_r)
    kwargs_source_mag_g[0]['magnitude'] -= 1

    # i-band
    kwargs_lens_light_mag_i = deepcopy(kwargs_lens_light_mag_r)
    kwargs_lens_light_mag_i[0]['magnitude'] += 1

    kwargs_source_mag_i = deepcopy(kwargs_source_mag_r)
    kwargs_source_mag_i[0]['magnitude'] += 1

    return [kwargs_lens_light_mag_g, kwargs_lens_light_mag_r, kwargs_lens_light_mag_i], [kwargs_source_mag_g, kwargs_source_mag_r, kwargs_source_mag_i]


def simulate_rgb(lens, kwargs_lens_list, kwargs_source_list, config_list, size):  
    if len(config_list) == 3:
        band_b, band_g, band_r = config_list
        kwargs_b_band = band_b.kwargs_single_band()
        kwargs_g_band = band_g.kwargs_single_band()
        kwargs_r_band = band_r.kwargs_single_band()
    elif len(config_list) == 1:
        band_g = config_list[0]
        kwargs_b_band = band_g.kwargs_single_band()
        kwargs_g_band = band_g.kwargs_single_band()
        kwargs_r_band = band_g.kwargs_single_band()
    else:
        raise Exception('Unknown telescope configuration')
    
    # set number of pixels from pixel scale
    pixel_scale = kwargs_g_band['pixel_scale']
    numpix = int(round(size / pixel_scale))

    sim_b = SimAPI(numpix=numpix, kwargs_single_band=kwargs_b_band, kwargs_model=lens.kwargs_model)
    sim_g = SimAPI(numpix=numpix, kwargs_single_band=kwargs_g_band, kwargs_model=lens.kwargs_model)
    sim_r = SimAPI(numpix=numpix, kwargs_single_band=kwargs_r_band, kwargs_model=lens.kwargs_model)

    kwargs_numerics = {}
    imSim_b = sim_b.image_model_class(kwargs_numerics)
    imSim_g = sim_g.image_model_class(kwargs_numerics)
    imSim_r = sim_r.image_model_class(kwargs_numerics)

    kwargs_lens_light_mag_g, kwargs_lens_light_mag_r, kwargs_lens_light_mag_i = kwargs_lens_list
    kwargs_source_mag_g, kwargs_source_mag_r, kwargs_source_mag_i = kwargs_source_list

    # turn magnitude kwargs into lenstronomy kwargs
    kwargs_lens_light_g, kwargs_source_g, _ = sim_b.magnitude2amplitude(kwargs_lens_light_mag_g, kwargs_source_mag_g)
    kwargs_lens_light_r, kwargs_source_r, _ = sim_g.magnitude2amplitude(kwargs_lens_light_mag_r, kwargs_source_mag_r)
    kwargs_lens_light_i, kwargs_source_i, _ = sim_r.magnitude2amplitude(kwargs_lens_light_mag_i, kwargs_source_mag_i)

    image_b = imSim_b.image(lens.kwargs_lens, kwargs_source_g, kwargs_lens_light_g)
    image_g = imSim_g.image(lens.kwargs_lens, kwargs_source_r, kwargs_lens_light_r)
    image_r = imSim_r.image(lens.kwargs_lens, kwargs_source_i, kwargs_lens_light_i)

    # add noise
    image_b += sim_b.noise_for_model(model=image_b)
    image_g += sim_g.noise_for_model(model=image_g)
    image_r += sim_r.noise_for_model(model=image_r)

    scale_max = 10000

    if len(config_list) == 3:
        img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
        img[:,:,0] = image_b  # _scale_max(image_b)
        img[:,:,1] = image_g  # _scale_max(image_g)
        img[:,:,2] = image_r  # _scale_max(image_r)
        # plot_util.sqrt(image_r, scale_min=0, scale_max=scale_max)
        data_class = sim_b.data_class

        return img, data_class
    elif len(config_list) == 1:
        img = image_g  # _scale_max(image_g)
        data_class = sim_b.data_class

        return img, data_class
    else:
        raise Exception('Unknown telescope configuration')
    

def _scale_max(image): 
    flat=image.flatten()
    flat.sort()
    scale_max = flat[int(len(flat)*0.95)]
    return scale_max


def lsst_config_list():
    LSST_g = LSST.LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
    LSST_r = LSST.LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
    LSST_i = LSST.LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)
    return [LSST_g, LSST_r, LSST_i]


def roman_rgb_config_list():
    Roman_g = Roman.Roman(band='F062', psf_type='PIXEL', survey_mode='wide_area')
    Roman_r = Roman.Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')
    Roman_i = Roman.Roman(band='F184', psf_type='PIXEL', survey_mode='wide_area')
    return [Roman_g, Roman_r, Roman_i]


def roman_config_list():
    return [Roman.Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')]


def des_config_list():
    DES_g = DES.DES(band='g', psf_type='GAUSSIAN', coadd_years=3)
    DES_r = DES.DES(band='r', psf_type='GAUSSIAN', coadd_years=3)
    DES_i = DES.DES(band='i', psf_type='GAUSSIAN', coadd_years=3)
    return [DES_g, DES_r, DES_i]


def hst_config_list():
    return [HST.HST(band='WFC3_F160W', psf_type='PIXEL', coadd_years=None)]


def euclid_config_list():
    return [Euclid.Euclid(band='VIS', psf_type='GAUSSIAN', coadd_years=6)]