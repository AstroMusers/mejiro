import os
from copy import deepcopy

import numpy as np
import pandas as pd
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.ObservationConfig import Roman
from lenstronomy.SimulationAPI.ObservationConfig import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import data_util
from lenstronomy.Util import util as len_util
from tqdm import tqdm

import mejiro
from mejiro.helpers.roman_params import RomanParameters

# get Roman params
module_path = os.path.dirname(mejiro.__file__)
csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
roman_params = RomanParameters(csv_path)


def get_snr(gglens, band, mask_mult=1., side=4.95, **kwargs):
    if 'total_image' and 'source_surface_brightness' and 'sim_api' in kwargs:
        total_image = kwargs['total_image']
        source = kwargs['source_surface_brightness']
        sim_api = kwargs['sim_api']
    else:
        total_image, _, source, sim_api = get_image(gglens, band, side=side)

    # calculate threshold that will define masked region
    stdev = np.std(source)
    mean = np.mean(source)
    threshold = mean + (mask_mult * stdev)

    # mask source
    masked_source = np.ma.masked_where(source < threshold, source)
    source_counts = masked_source.compressed().sum()

    # add noise
    noise = sim_api.noise_for_model(model=total_image)
    total_image_with_noise = total_image + noise

    # mask total image
    masked_total_image = np.ma.array(mask=masked_source.mask, data=total_image)
    total_counts = masked_total_image.compressed().sum()

    # calculate estimated SNR
    snr = source_counts / np.sqrt(total_counts)

    return snr, total_image_with_noise


def get_image(gglens, band, side=4.95):
    kwargs_model, kwargs_params = gglens.lenstronomy_kwargs(band=band)

    Roman_r = Roman.Roman(band=band.upper(), psf_type='PIXEL', survey_mode='wide_area')
    Roman_r.obs['num_exposures'] = 1  # set number of exposures to 1 cf. 96
    kwargs_r_band = Roman_r.kwargs_single_band()

    sim_r = SimAPI(numpix=int(side / 0.11), kwargs_single_band=kwargs_r_band, kwargs_model=kwargs_model)

    kwargs_numerics = {'point_source_supersampling_factor': 1, 'supersampling_factor': 3}
    imSim_r = sim_r.image_model_class(kwargs_numerics)

    kwargs_lens = kwargs_params['kwargs_lens']
    kwargs_lens_light = kwargs_params['kwargs_lens_light']
    kwargs_source = kwargs_params['kwargs_source']

    kwargs_lens_light_r, kwargs_source_r, _ = sim_r.magnitude2amplitude(kwargs_lens_light, kwargs_source)

    # set lens light to 0 for source image
    source_kwargs_lens_light_r = deepcopy(kwargs_lens_light_r)
    source_kwargs_lens_light_r[0]['amp'] = 0
    source_surface_brightness = imSim_r.image(kwargs_lens, kwargs_source_r, source_kwargs_lens_light_r, None)

    # set source light to 0 for lens image
    lens_kwargs_source_r = deepcopy(kwargs_source_r)
    lens_kwargs_source_r[0]['amp'] = 0
    lens_surface_brightness = imSim_r.image(kwargs_lens, lens_kwargs_source_r, kwargs_lens_light_r, None)

    total_image = imSim_r.image(kwargs_lens, kwargs_source_r, kwargs_lens_light_r, None)

    return total_image, lens_surface_brightness, source_surface_brightness, sim_r


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
