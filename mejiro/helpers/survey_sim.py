import os

import numpy as np
import pandas as pd
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.ObservationConfig import Roman
from lenstronomy.Util import data_util, util
from tqdm import tqdm


from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.ObservationConfig import Roman
from lenstronomy.Util import data_util, util


def get_snr_hwo(gglens, band):
    total_image, lens_surface_brightness, source_surface_brightness = get_image_hwo(gglens, band)

    # calculate region for source surface brightness array and count signal
    stdev = np.std(source_surface_brightness)
    mean = np.mean(source_surface_brightness)
    mask = source_surface_brightness < mean + (1 * stdev)
    masked_source = np.ma.masked_array(source_surface_brightness, mask=mask)
    sum_source_counts = np.sum(masked_source)

    # estimate and add background
    min_zodiacal_light_f106 = 0.29
    thermal_background_f106 = 0
    estimated_background = (min_zodiacal_light_f106 * 1.5) + thermal_background_f106  # in counts/pixel
    estimated_background *= np.ones(total_image.shape)
    total_image += estimated_background

    # estimated total noise in 180s from https://roman.gsfc.nasa.gov/science/WFI_technical.html
    noise_180s = 5.5  # e- RMS
    noise_rms = noise_180s * (146 / 180)
    noise_array = np.random.poisson(noise_rms, total_image.shape)
    total_image += noise_array

    # count total signal
    masked_total = np.ma.masked_array(total_image, mask=mask)
    sum_total_counts = np.sum(masked_total)

    # calculate estimated SNR
    return sum_source_counts / np.sqrt(sum_total_counts), total_image


def get_image_hwo(gglens, band):
    kwargs_model, kwargs_params = gglens.lenstronomy_kwargs(band=band)

    magnitude_zero_point = 28.7339
    lens_model = LensModel(kwargs_model['lens_model_list'])
    source_light_model = LightModel(kwargs_model['source_light_model_list'])
    lens_light_model = LightModel(kwargs_model['lens_light_model_list'])
    kwargs_lens = kwargs_params['kwargs_lens']
    kwargs_source = kwargs_params['kwargs_source']
    kwargs_lens_light = kwargs_params['kwargs_lens_light']
    kwargs_source_amp = data_util.magnitude2amplitude(source_light_model, kwargs_source, magnitude_zero_point)
    kwargs_lens_light_amp = data_util.magnitude2amplitude(lens_light_model, kwargs_lens_light, magnitude_zero_point)

    # set up image
    num_pix = 720
    psf_class = PSF(**{'psf_type': 'GAUSSIAN', 'fwhm': 0.0021})
    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(
        numPix=num_pix,
        deltapix=0.00687549354156988,
        subgrid_res=1,
        left_lower=False,
        inverse=False)
    kwargs_pixel = {'nx': num_pix, 'ny': num_pix,  # number of pixels per axis
                    'ra_at_xy_0': ra_at_xy_0,
                    'dec_at_xy_0': dec_at_xy_0,
                    'transform_pix2angle': Mpix2coord}
    pixel_grid = PixelGrid(**kwargs_pixel)
    image_model = ImageModel(data_class=pixel_grid,
                             psf_class=psf_class,
                             lens_model_class=lens_model,
                             source_model_class=source_light_model,
                             lens_light_model_class=lens_light_model,
                             kwargs_numerics=kwargs_numerics)

    # get surface brightness arrays
    lens_surface_brightness = image_model.lens_surface_brightness(kwargs_lens_light_amp)
    source_surface_brightness = image_model.source_surface_brightness(kwargs_source_amp, kwargs_lens)
    total_image = image_model.image(kwargs_lens=kwargs_lens,
                                    kwargs_source=kwargs_source_amp,
                                    kwargs_lens_light=kwargs_lens_light_amp)

    return total_image, lens_surface_brightness, source_surface_brightness


def get_snr(gglens, band):
    total_image, lens_surface_brightness, source_surface_brightness = get_image(gglens, band)

    # calculate region for source surface brightness array and count signal
    stdev = np.std(source_surface_brightness)
    mean = np.mean(source_surface_brightness)
    mask = source_surface_brightness < mean + (1 * stdev)
    masked_source = np.ma.masked_array(source_surface_brightness, mask=mask)
    sum_source_counts = np.sum(masked_source)

    # estimate and add background
    min_zodiacal_light_f106 = 0.29
    thermal_background_f106 = 0
    estimated_background = (min_zodiacal_light_f106 * 1.5) + thermal_background_f106  # in counts/pixel
    estimated_background *= np.ones(total_image.shape)
    total_image += estimated_background

    # estimated total noise in 180s from https://roman.gsfc.nasa.gov/science/WFI_technical.html
    noise_180s = 5.5  # e- RMS
    noise_rms = noise_180s * (146 / 180)
    noise_array = np.random.poisson(noise_rms, total_image.shape)
    total_image += noise_array

    # count total signal
    masked_total = np.ma.masked_array(total_image, mask=mask)
    sum_total_counts = np.sum(masked_total)

    # calculate estimated SNR
    return sum_source_counts / np.sqrt(sum_total_counts), total_image


def get_image(gglens, band):
    kwargs_model, kwargs_params = gglens.lenstronomy_kwargs(band=band)

    magnitude_zero_point = Roman.F106_band_obs['magnitude_zero_point']
    lens_model = LensModel(kwargs_model['lens_model_list'])
    source_light_model = LightModel(kwargs_model['source_light_model_list'])
    lens_light_model = LightModel(kwargs_model['lens_light_model_list'])
    kwargs_lens = kwargs_params['kwargs_lens']
    kwargs_source = kwargs_params['kwargs_source']
    kwargs_lens_light = kwargs_params['kwargs_lens_light']
    kwargs_source_amp = data_util.magnitude2amplitude(source_light_model, kwargs_source, magnitude_zero_point)
    kwargs_lens_light_amp = data_util.magnitude2amplitude(lens_light_model, kwargs_lens_light, magnitude_zero_point)

    # set up image
    num_pix = 45  # at 0.11 pixels/", 10.01"
    psf_class = PSF(**{'psf_type': 'GAUSSIAN', 'fwhm': 0.087})
    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(
        numPix=num_pix,
        deltapix=0.11,
        subgrid_res=1,
        left_lower=False,
        inverse=False)
    kwargs_pixel = {'nx': num_pix, 'ny': num_pix,  # number of pixels per axis
                    'ra_at_xy_0': ra_at_xy_0,
                    'dec_at_xy_0': dec_at_xy_0,
                    'transform_pix2angle': Mpix2coord}
    pixel_grid = PixelGrid(**kwargs_pixel)
    image_model = ImageModel(data_class=pixel_grid,
                             psf_class=psf_class,
                             lens_model_class=lens_model,
                             source_model_class=source_light_model,
                             lens_light_model_class=lens_light_model,
                             kwargs_numerics=kwargs_numerics)

    # get surface brightness arrays
    lens_surface_brightness = image_model.lens_surface_brightness(kwargs_lens_light_amp)
    source_surface_brightness = image_model.source_surface_brightness(kwargs_source_amp, kwargs_lens)
    total_image = image_model.image(kwargs_lens=kwargs_lens,
                                    kwargs_source=kwargs_source_amp,
                                    kwargs_lens_light=kwargs_lens_light_amp)

    return total_image, lens_surface_brightness, source_surface_brightness


def write_lens_pop_to_csv(output_path, gg_lenses, bands, suppress_output=True):
    dictparaggln = {}
    dictparaggln['Candidate'] = {}
    listnamepara = ['velodisp', 'massstel', 'angleins', 'redssour', 'redslens', 'magnsour', 'numbimag',
                    'maxmdistimag']  # 'xposlens', 'yposlens', 'xpossour', 'ypossour',
    for nameband in bands:
        listnamepara += ['magtlens%s' % nameband]
        listnamepara += ['magtsour%s' % nameband]
        listnamepara += ['magtsourMagnified%s' % nameband]

    for namepara in listnamepara:
        dictparaggln['Candidate'][namepara] = np.empty(len(gg_lenses))

    df = pd.DataFrame(columns=listnamepara)

    for i, gg_lens in tqdm(enumerate(gg_lenses), total=len(gg_lenses), disable=suppress_output):
        dict = {
            'velodisp': gg_lens.deflector_velocity_dispersion(),
            'massstel': gg_lens.deflector_stellar_mass() * 1e-12,
            'angleins': gg_lens.einstein_radius,
            'redssour': gg_lens.source_redshift,
            'redslens': gg_lens.deflector_redshift,
            'magnsour': gg_lens.extended_source_magnification()
            # TODO add SNR?
        }

        posiimag = gg_lens.point_source_image_positions()
        dict['numbimag'] = int(posiimag[0].size)

        dict['maxmdistimag'] = np.amax(np.sqrt(
            (posiimag[0][:, None] - posiimag[0][None, :]) ** 2 + (posiimag[1][:, None] - posiimag[1][None, :]) ** 2))

        # TODO ypossour was throwing index 1 out of bounds. but I also don't need this info (for now) so maybe just delete
        # posilens = gg_lens.deflector_position
        # posisour = gg_lens.extended_source_image_positions()[0]
        # dict['xposlens'] = posilens[0]
        # dict['yposlens'] = posilens[1]
        # dict['xpossour'] = posisour[0]
        # dict['ypossour'] = posisour[1]

        for nameband in bands:
            dict['magtlens%s' % nameband] = gg_lens.deflector_magnitude(band=nameband)
            dict['magtsour%s' % nameband] = gg_lens.extended_source_magnitude(band=nameband)
            dict['magtsourMagnified%s' % nameband] = gg_lens.extended_source_magnitude(band=nameband, lensed=True)

        df.loc[i] = pd.Series(dict)

    if not suppress_output: print('Writing to %s..' % output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    df.to_csv(output_path, index=False)
