import numpy as np
import os
import pandas as pd
from astropy.cosmology import default_cosmology
from copy import deepcopy
from lenstronomy.SimulationAPI.ObservationConfig import Roman
from lenstronomy.SimulationAPI.sim_api import SimAPI
from tqdm import tqdm
from scipy import ndimage

import mejiro
from mejiro.analysis import regions
from mejiro.helpers import gs
from mejiro.helpers.roman_params import RomanParameters
from mejiro.lenses import lens_util
from mejiro.lenses.strong_lens import StrongLens
from mejiro.plots import diagnostic_plot

# get Roman params
module_path = os.path.dirname(mejiro.__file__)
csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
roman_params = RomanParameters(csv_path)


# TODO a(n imperfect) lens subtraction option?
def get_snr(gglens, band, zp, detector=1, detector_position=(2048, 2048), input_num_pix=51, output_num_pix=45,
            side=4.95, oversample=1,
            exposure_time=146, add_subhalos=True,
            kwargs_numerics={'supersampling_factor': 3, 'compute_mode': 'regular'}, return_snr_list=False,
            debugging=False, debug_dir=None,
            psf_cache_dir=None, snr_per_pixel_threshold=1):
    if debugging: assert debug_dir is not None, 'Debugging is enabled but no debug directory is provided.'

    if type(detector) is str:
        detector = int(detector)

    check_cache = False if psf_cache_dir is None else True

    if type(gglens) is not StrongLens:
        strong_lens = lens_util.slsim_lens_to_mejiro(gglens, bands=[band],
                                                     cosmo=default_cosmology.get())  # TODO pass in cosmology
    else:
        strong_lens = gglens

    if add_subhalos:
        realization = strong_lens.generate_cdm_subhalos()
        strong_lens.add_subhalos(realization)

    # generate synthetic images with lenstronomy
    strong_lens.num_pix = input_num_pix * oversample  # TODO temporary workaround
    strong_lens.side = side  # TODO temporary workaround
    supersampling_indices = strong_lens.build_adaptive_grid(input_num_pix * oversample, pad=25)
    if supersampling_indices is None:
        return None, None, None, None
    kwargs_numerics_TEMP = {
        'supersampling_factor': kwargs_numerics['supersampling_factor'],
        'compute_mode': kwargs_numerics['compute_mode'],
        'supersampled_indexes': supersampling_indices
    }
    model, lens_sb, source_sb = strong_lens.get_array(input_num_pix * oversample, side, band, zp,
                                                      kwargs_numerics=kwargs_numerics_TEMP, return_pieces=True)

    # generate GalSim images
    results, lenses, sources, _ = gs.get_images(strong_lens, [model], [band], {band: zp}, input_num_pix, output_num_pix,
                                                oversample,
                                                oversample,
                                                lens_surface_brightness=[lens_sb],
                                                source_surface_brightness=[source_sb], detector=detector,
                                                detector_pos=detector_position,
                                                exposure_time=exposure_time, ra=30, dec=-30, seed=None, validate=False,
                                                suppress_output=True, check_cache=check_cache,
                                                psf_cache_dir=psf_cache_dir)

    # put back into units of counts
    total = results[0] * exposure_time
    lens = lenses[0] * exposure_time
    source = sources[0] * exposure_time

    # get noise; NB neither lens or source has sky background to detector effects added
    noise = total - (lens + source)

    # calculate SNR in each pixel
    snr_array = np.nan_to_num(source / np.sqrt(total))

    if not np.any(snr_array >= snr_per_pixel_threshold):
        return 1, np.ma.array(snr_array, mask=True), [
            1], None  # TODO I'm doing this because SNRs for non-detectable lenses don't really matter right now
        # masked_snr_array = np.ma.masked_where(snr_array <= np.quantile(snr_array, 0.9), snr_array)  # TODO this must be the mask that's causing issues
    else:
        masked_snr_array = np.ma.masked_where(snr_array <= snr_per_pixel_threshold, snr_array)

    # # speed up region-identifying code by removing single pixel regions
    # masked_snr_array = regions.remove_single_pixels(masked_snr_array)

    # # calculate regions of connected pixels given the snr mask
    # indices_list = regions.get_regions(masked_snr_array, debug_dir)

    # if indices_list is None:
    #     return None, None, None, None

    # snr_list = []
    # for region in indices_list:
    #     numerator, denominator = 0, 0
    #     for j, i in region:
    #         numerator += source[i, j]
    #         denominator += source[i, j] + lens[i, j] + noise[i, j]
    #     if denominator <= 0.:
    #         continue
    #     snr = numerator / np.sqrt(denominator)
    #     assert not np.isnan(snr), 'NaN in SNR list'
    #     assert not np.isinf(snr), 'Inf in SNR list'
    #     snr_list.append(snr)

    structure = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])
    labeled_array, num_regions = ndimage.label(masked_snr_array.filled(0), structure=structure)

    # calculate the SNR for each region
    snr_list = []
    for i in range(1, num_regions + 1):
        source_counts = np.sum(source[labeled_array == i])
        total_counts = np.sum(total[labeled_array == i])
        snr = source_counts / np.sqrt(total_counts)
        snr_list.append(snr)

    if len(snr_list) != 0:
        if debugging and np.max(snr_list) > 20:
            diagnostic_plot.snr_plot(labeled_array, strong_lens, total, lens, source, noise, snr_array, masked_snr_array, snr_list,
                                     debug_dir)

    if return_snr_list:
        return snr_list, None, None, None
    else:
        if len(snr_list) != 0:
            return np.max(snr_list), masked_snr_array, snr_list, None
        else:
            return None, None, None, None


def get_snr_lenstronomy(gglens, band, subtract_lens=True, mask_mult=1., side=4.95, **kwargs):
    if 'total_image' and 'source_surface_brightness' and 'sim_api' in kwargs:
        total_image = kwargs['total_image']
        source = kwargs['source_surface_brightness']
        sim_api = kwargs['sim_api']
    else:
        total_image, lens, source, sim_api = get_image(gglens, band, side=side)

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

    if subtract_lens:
        total_image_with_noise -= lens

    # mask total image
    masked_total_image = np.ma.array(mask=masked_source.mask, data=total_image_with_noise)
    total_counts = masked_total_image.compressed().sum()

    # calculate estimated SNR
    with np.errstate(invalid='raise'):
        try:
            snr = source_counts / np.sqrt(total_counts)
        except:
            snr = 0

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


def write_lens_pop_to_csv(output_path, gg_lenses, detectable_snr_list, bands, suppress_output=True):
    dictparaggln = {}
    dictparaggln['Candidate'] = {}
    listnamepara = ['velodisp', 'massstel', 'angleins', 'redssour', 'redslens', 'magnsour', 'snr', 'numbimag',
                    'maxmdistimag']  # 'xposlens', 'yposlens', 'xpossour', 'ypossour',
    for nameband in bands:
        listnamepara += ['magtlens%s' % nameband]
        listnamepara += ['magtsour%s' % nameband]
        listnamepara += ['magtsourMagnified%s' % nameband]

    for namepara in listnamepara:
        dictparaggln['Candidate'][namepara] = np.empty(len(gg_lenses))

    df = pd.DataFrame(columns=listnamepara)

    for i, (gg_lens, snr) in tqdm(enumerate(zip(gg_lenses, detectable_snr_list)), total=len(gg_lenses),
                                  disable=suppress_output):
        dict = {
            'velodisp': gg_lens.deflector_velocity_dispersion(),
            'massstel': gg_lens.deflector_stellar_mass() * 1e-12,
            'angleins': gg_lens.einstein_radius,
            'redssour': gg_lens.source_redshift,
            'redslens': gg_lens.deflector_redshift,
            'magnsour': gg_lens.extended_source_magnification(),
            'snr': snr
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
            dict['magtsour%s' % nameband] = gg_lens.extended_source_magnitude(band=nameband, lensed=False)
            dict['magtsourMagnified%s' % nameband] = gg_lens.extended_source_magnitude(band=nameband, lensed=True)

        df.loc[i] = pd.Series(dict)

    if not suppress_output: print('Writing to %s..' % output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    df.to_csv(output_path, index=False)
