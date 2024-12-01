import warnings
from copy import deepcopy

import galsim
import galsim.roman  # NB not automatically imported with `import galsim`
import numpy as np

from mejiro.utils import roman_util


def default_roman_engine_params():
    """
    Returns the default parameters for Roman image simulation with the GalSim engine.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        - rng: galsim.UniformDeviate(42)
        - sky_background: bool, default is True
        - detector_effects: bool, default is True
        - poisson_noise: bool, default is True
        - reciprocity_failure: bool, default is True
        - dark_noise: bool, default is True
        - nonlinearity: bool, default is True
        - ipc: bool, default is True
        - read_noise: bool, default is True
    """
    return {
        'rng': galsim.UniformDeviate(42),
        'sky_background': True,
        'detector_effects': True,
        'poisson_noise': True,
        'reciprocity_failure': True,
        'dark_noise': True,
        'nonlinearity': True,
        'ipc': True,
        'read_noise': True,
    }


def default_hwo_engine_params():
    """
    Returns the default parameters for HWO image simulation with the GalSim engine.

    Returns
    -------
    dict
        A dictionary containing the following keys:

        - rng: galsim.UniformDeviate(42)
        - sky_background: bool, default is True
        - detector_effects: bool, default is True
        - poisson_noise: bool, default is True
        - dark_noise: bool, default is True
        - read_noise: bool, default is True
    """
    return {
        'rng': galsim.UniformDeviate(42),
        'sky_background': True,
        'detector_effects': True,
        'poisson_noise': True,
        'dark_noise': True,
        'read_noise': True,
    }


def get_roman_exposure(synthetic_image, exposure_time, psf=None, engine_params=default_roman_engine_params(),
                       verbose=False, **kwargs):
    # get detector and detector position
    detector = synthetic_image.instrument_params['detector']
    detector_position = synthetic_image.instrument_params['detector_position']

    # get optional kwargs
    gsparams_kwargs = kwargs.get('gsparams_kwargs', {})

    # create interpolated image
    total_interp = galsim.InterpolatedImage(galsim.Image(synthetic_image.image, xmin=0, ymin=0),
                                            scale=synthetic_image.pixel_scale,
                                            flux=np.sum(synthetic_image.image) * exposure_time)

    # get PSF
    if psf is None:
        from mejiro.engines import webbpsf_engine

        check_cache = kwargs.get('check_cache', True)
        psf_cache_dir = kwargs.get('psf_cache_dir', '/data/bwedig/mejiro/cached_psfs')
        calc_psf_kwargs = kwargs.get('calc_psf_kwargs', {})

        psf = webbpsf_engine.get_roman_psf(band=synthetic_image.band,
                                           detector=detector,
                                           detector_position=detector_position,
                                           oversample=synthetic_image.oversample,
                                           num_pix=101,  # NB WebbPSF wants the native pixel size
                                           check_cache=check_cache,
                                           psf_cache_dir=psf_cache_dir,
                                           verbose=verbose,
                                           **calc_psf_kwargs)
    psf_interp = galsim.InterpolatedImage(galsim.Image(psf, scale=synthetic_image.pixel_scale))

    # convolve with PSF
    convolved = galsim.Convolve(total_interp, psf_interp, gsparams=galsim.GSParams(**gsparams_kwargs))

    # draw image at the native pixel scale
    im = galsim.ImageF(synthetic_image.native_num_pix, synthetic_image.native_num_pix,
                       scale=synthetic_image.native_pixel_scale)
    im.setOrigin(0, 0)
    image = convolved.drawImage(im)

    # NB from here on out, the image is at the native pixel scale, i.e., the image is NOT oversampled

    # add sky background
    if engine_params['sky_background']:
        bkgs = get_roman_sky_background(synthetic_image.instrument, synthetic_image.band, detector, exposure_time,
                                        num_pix=synthetic_image.native_num_pix, oversample=1)
        bkg = bkgs[synthetic_image.band]
        image += bkg

    # integer number of photons are being detected, so quantize
    image.quantize()

    # add detector effects
    if engine_params['detector_effects']:
        image.replaceNegative(0.)

        # Poisson noise
        if type(engine_params['poisson_noise']) is galsim.Image:
            poisson_noise = engine_params['poisson_noise']
            image += poisson_noise
            image.quantize()
        elif type(engine_params['poisson_noise']) is bool:
            if engine_params['poisson_noise']:
                before = deepcopy(image)
                image.addNoise(galsim.PoissonNoise(engine_params['rng']))
                poisson_noise = image - before
                image.quantize()
            else:
                poisson_noise = None

        # reciprocity failure
        if type(engine_params['reciprocity_failure']) is galsim.Image:
            reciprocity_failure = engine_params['reciprocity_failure']
            image += reciprocity_failure
        elif type(engine_params['reciprocity_failure']) is bool:
            if engine_params['reciprocity_failure']:
                before = deepcopy(image)
                galsim.roman.addReciprocityFailure(image, exptime=exposure_time)
                reciprocity_failure = image - before
            else:
                reciprocity_failure = None

        # dark current
        if type(engine_params['dark_noise']) is galsim.Image:
            dark_noise = engine_params['dark_noise']
            image += dark_noise
        elif type(engine_params['dark_noise']) is bool:
            if engine_params['dark_noise']:
                before = deepcopy(image)
                total_dark_current = galsim.roman.dark_current * exposure_time
                image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(engine_params['rng'], total_dark_current)))
                dark_noise = image - before
            else:
                dark_noise = None

        # skip persistence because we're simulating a single exposure

        # nonlinearity
        if type(engine_params['nonlinearity']) is galsim.Image:
            nonlinearity = engine_params['nonlinearity']
            image += nonlinearity
        elif type(engine_params['nonlinearity']) is bool:
            if engine_params['nonlinearity']:
                before = deepcopy(image)
                galsim.roman.applyNonlinearity(image)
                nonlinearity = image - before
            else:
                nonlinearity = None

        # IPC
        if type(engine_params['ipc']) is galsim.Image:
            ipc = engine_params['ipc']
            image += ipc
        elif type(engine_params['ipc']) is bool:
            if engine_params['ipc']:
                before = deepcopy(image)
                galsim.roman.applyIPC(image)
                ipc = image - before
            else:
                ipc = None

        # read noise
        if type(engine_params['read_noise']) is galsim.Image:
            read_noise = engine_params['read_noise']
            image += read_noise
        elif type(engine_params['read_noise']) is bool:
            if engine_params['read_noise']:
                before = deepcopy(image)
                read_noise_sigma = galsim.roman.read_noise
                image.addNoise(galsim.GaussianNoise(engine_params['rng'], sigma=read_noise_sigma))
                read_noise = image - before
            else:
                read_noise = None

        # gain
        image /= galsim.roman.gain

        # quantize, since analog-to-digital conversion gives integers
        image.quantize()

        # if any unphysical negative pixels exist due to how GalSim adds Poisson noise, set them to zero
        if np.any(image.array < 0):
            warnings.warn('Negative pixel values in final image')
            image.replaceNegative(0.)
    else:
        poisson_noise = None
        reciprocity_failure = None
        dark_noise = None
        nonlinearity = None
        ipc = None
        read_noise = None

    if synthetic_image.pieces:
        lens_image = get_piece(synthetic_image.lens_surface_brightness, synthetic_image.pixel_scale,
                               synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                               psf_interp, gsparams_kwargs)
        source_image = get_piece(synthetic_image.source_surface_brightness, synthetic_image.pixel_scale,
                                 synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                                 psf_interp, gsparams_kwargs)
        results = image, lens_image, source_image
    else:
        results = image

    return results, psf, poisson_noise, reciprocity_failure, dark_noise, nonlinearity, ipc, read_noise


def get_piece(surface_brightness, pixel_scale, native_pixel_scale, native_num_pix, exposure_time, psf_interp,
              gsparams_kwargs):
    interp = galsim.InterpolatedImage(galsim.Image(surface_brightness, xmin=0, ymin=0), scale=pixel_scale,
                                      flux=np.sum(surface_brightness) * exposure_time)

    convolved = galsim.Convolve(interp, psf_interp, gsparams=galsim.GSParams(**gsparams_kwargs))

    im = galsim.ImageF(native_num_pix, native_num_pix, scale=native_pixel_scale)
    im.setOrigin(0, 0)
    image = convolved.drawImage(im)
    image.replaceNegative(0.)
    image.quantize()

    return image


def get_roman_sky_background(instrument, bands, sca, exposure_time, num_pix, oversample):
    # was only one band provided as a string? or a list of bands?
    if not isinstance(bands, list):
        bands = [bands]

    bkgs = {}
    for band in bands:
        # build Image
        sky_image = galsim.ImageF(num_pix, num_pix)

        # get minimum zodiacal light in this band in counts/pixel/sec
        sky_level = instrument.get_min_zodi(band, sca)

        # "For observations at high galactic latitudes, the Zodi intensity is typically ~1.5x the minimum" (https://roman.gsfc.nasa.gov/science/WFI_technical.html)
        sky_level *= 1.5

        # the stray light level is currently set in GalSim to a pessimistic 10% of sky level
        sky_level *= (1. + galsim.roman.stray_light_fraction)

        # get thermal background in this band in counts/pixel/sec
        thermal_bkg = instrument.get_thermal_bkg(band, sca)

        # combine the two backgrounds (still counts/pixel/sec)
        sky_image += sky_level
        sky_image += thermal_bkg

        # convert to counts/pixel
        sky_image *= exposure_time

        # if the image is oversampled, the sky background must be spread out over more pixels
        sky_image /= oversample ** 2

        bkgs[band] = sky_image

    return bkgs


def get_hwo_exposure(synthetic_image, exposure_time, psf=None, engine_params=default_hwo_engine_params(), verbose=False,
                     **kwargs):
    # get optional kwargs
    gsparams_kwargs = kwargs.get('gsparams_kwargs', {})

    # create interpolated image
    total_interp = galsim.InterpolatedImage(galsim.Image(synthetic_image.image, xmin=0, ymin=0),
                                            scale=synthetic_image.pixel_scale,
                                            flux=np.sum(synthetic_image.image) * exposure_time)

    # get PSF
    if psf is None:
        psf_interp = get_gaussian_psf(synthetic_image.instrument.get_psf_fwhm(synthetic_image.band))
        psf = psf_interp.drawImage(nx=synthetic_image.native_num_pix, ny=synthetic_image.native_num_pix,
                                   scale=synthetic_image.native_pixel_scale)

    # convolve with PSF
    convolved = galsim.Convolve(total_interp, psf_interp, gsparams=galsim.GSParams(**gsparams_kwargs))

    # draw image at the native pixel scale
    im = galsim.ImageF(synthetic_image.native_num_pix, synthetic_image.native_num_pix,
                       scale=synthetic_image.native_pixel_scale)
    im.setOrigin(0, 0)
    image = convolved.drawImage(im)

    # NB from here on out, the image is at the native pixel scale, i.e., the image is NOT oversampled

    # add sky background
    if engine_params['sky_background']:
        min_zodi_cps = 0.2
        sky_bkg_cps = min_zodi_cps * 1.5

        # build Image
        sky_image = galsim.ImageF(synthetic_image.native_num_pix, synthetic_image.native_num_pix)
        sky_image.setOrigin(0, 0)
        sky_image += sky_bkg_cps

        # convert to counts/pixel
        sky_image *= exposure_time

        image += sky_image

    # integer number of photons are being detected, so quantize
    image.quantize()

    if engine_params['detector_effects']:
        image.replaceNegative(0.)

        # Poisson noise
        if type(engine_params['poisson_noise']) is galsim.Image:
            poisson_noise = engine_params['poisson_noise']
            image += poisson_noise
            image.quantize()
        elif type(engine_params['poisson_noise']) is bool:
            if engine_params['poisson_noise']:
                before = deepcopy(image)
                image.addNoise(galsim.PoissonNoise(engine_params['rng']))
                poisson_noise = image - before
                image.quantize()
            else:
                poisson_noise = None

        # dark current
        if type(engine_params['dark_noise']) is galsim.Image:
            dark_noise = engine_params['dark_noise']
            image += dark_noise
        elif type(engine_params['dark_noise']) is bool:
            if engine_params['dark_noise']:
                before = deepcopy(image)
                total_dark_current = synthetic_image.instrument.get_dark_current(synthetic_image.band) * exposure_time
                image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(engine_params['rng'], total_dark_current)))
                dark_noise = image - before
            else:
                dark_noise = None

        # read noise
        if type(engine_params['read_noise']) is galsim.Image:
            read_noise = engine_params['read_noise']
            image += read_noise
        elif type(engine_params['read_noise']) is bool:
            if engine_params['read_noise']:
                before = deepcopy(image)
                read_noise_sigma = synthetic_image.instrument.get_read_noise(synthetic_image.band)
                image.addNoise(galsim.GaussianNoise(engine_params['rng'], sigma=read_noise_sigma))
                read_noise = image - before
            else:
                read_noise = None

        # gain
        image /= synthetic_image.instrument.gain

        # quantize, since analog-to-digital conversion gives integers
        image.quantize()

        # if any unphysical negative pixels exists due to how GalSim adds Poisson noise, set them to zero
        if np.any(image.array < 0):
            warnings.warn('Negative pixel values in final image')
            image.replaceNegative(0.)
    else:
        poisson_noise = None
        dark_noise = None
        read_noise = None

    if synthetic_image.pieces:
        lens_image = get_piece(synthetic_image.lens_surface_brightness, synthetic_image.pixel_scale,
                               synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                               psf_interp, gsparams_kwargs)
        source_image = get_piece(synthetic_image.source_surface_brightness, synthetic_image.pixel_scale,
                                 synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                                 psf_interp, gsparams_kwargs)
        results = image, lens_image, source_image
    else:
        results = image

    return results, psf, poisson_noise, dark_noise, read_noise


def get_gaussian_psf(fwhm, flux=1.):
    return galsim.Gaussian(fwhm=fwhm, flux=flux)


def get_roman_psf(band, detector, detector_position, pupil_bin=1):
    return galsim.roman.getPSF(SCA=detector,
                               SCA_pos=galsim.PositionD(*detector_position),
                               bandpass=None,
                               wavelength=roman_util.translate_band(band),
                               pupil_bin=pupil_bin)


def validate_roman_engine_params(engine_params):
    if 'rng' not in engine_params.keys():
        engine_params['rng'] = default_roman_engine_params()['rng']  # TODO is this necessary? doesn't GalSim do this?
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'sky_background' not in engine_params.keys():
        engine_params['sky_background'] = default_roman_engine_params()['sky_background']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'detector_effects' not in engine_params.keys():
        engine_params['detector_effects'] = default_roman_engine_params()['detector_effects']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'poisson_noise' not in engine_params.keys():
        engine_params['poisson_noise'] = default_roman_engine_params()['poisson_noise']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'reciprocity_failure' not in engine_params.keys():
        engine_params['reciprocity_failure'] = default_roman_engine_params()['reciprocity_failure']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'dark_noise' not in engine_params.keys():
        engine_params['dark_noise'] = default_roman_engine_params()['dark_noise']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'nonlinearity' not in engine_params.keys():
        engine_params['nonlinearity'] = default_roman_engine_params()['nonlinearity']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'ipc' not in engine_params.keys():
        engine_params['ipc'] = default_roman_engine_params()['ipc']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'read_noise' not in engine_params.keys():
        engine_params['read_noise'] = default_roman_engine_params()['read_noise']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    return engine_params


def validate_hwo_engine_params(engine_params):
    if 'rng' not in engine_params.keys():
        engine_params['rng'] = default_hwo_engine_params()['rng']  # TODO is this necessary? doesn't GalSim do this?
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'sky_background' not in engine_params.keys():
        engine_params['sky_background'] = default_hwo_engine_params()['sky_background']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'detector_effects' not in engine_params.keys():
        engine_params['detector_effects'] = default_hwo_engine_params()['detector_effects']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'poisson_noise' not in engine_params.keys():
        engine_params['poisson_noise'] = default_hwo_engine_params()['poisson_noise']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'dark_noise' not in engine_params.keys():
        engine_params['dark_noise'] = default_hwo_engine_params()['dark_noise']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    if 'read_noise' not in engine_params.keys():
        engine_params['read_noise'] = default_hwo_engine_params()['read_noise']
        # TODO logging to inform user of default
    else:
        # TODO validate
        pass
    return engine_params


def get_empty_image(num_pix, pixel_scale):
    return galsim.ImageF(num_pix, num_pix, scale=pixel_scale)
