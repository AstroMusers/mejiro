import galsim
import galsim.roman  # NB not automatically imported with `import galsim`
import numpy as np
import warnings
from copy import deepcopy

from mejiro.engines.engine import Engine
from mejiro.utils import roman_util


class GalSimEngine(Engine):
    @staticmethod
    def defaults(instrument_name):
        if instrument_name.lower() == 'roman':
            return {
                'rng_seed': 42,
                'sky_background': True,
                'detector_effects': True,
                'poisson_noise': True,
                'reciprocity_failure': True,
                'dark_noise': True,
                'nonlinearity': True,
                'ipc': True,
                'read_noise': True,
            }
        elif instrument_name.lower() == 'hwo':
            return {
                'rng_seed': 42,
                'sky_background': True,
                'detector_effects': True,
                'poisson_noise': True,
                'dark_noise': True,
                'read_noise': True,
            }
        else:
            Engine.instrument_not_supported(instrument_name)


    @staticmethod
    def validate_engine_params(instrument_name, engine_params):
        if instrument_name.lower() == 'roman':
            if 'rng_seed' not in engine_params.keys():
                engine_params['rng_seed'] = GalSimEngine.defaults('Roman')['rng_seed']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'sky_background' not in engine_params.keys():
                engine_params['sky_background'] = GalSimEngine.defaults('Roman')['sky_background']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'detector_effects' not in engine_params.keys():
                engine_params['detector_effects'] = GalSimEngine.defaults('Roman')['detector_effects']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'poisson_noise' not in engine_params.keys():
                engine_params['poisson_noise'] = GalSimEngine.defaults('Roman')['poisson_noise']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'reciprocity_failure' not in engine_params.keys():
                engine_params['reciprocity_failure'] = GalSimEngine.defaults('Roman')['reciprocity_failure']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'dark_noise' not in engine_params.keys():
                engine_params['dark_noise'] = GalSimEngine.defaults('Roman')['dark_noise']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'nonlinearity' not in engine_params.keys():
                engine_params['nonlinearity'] = GalSimEngine.defaults('Roman')['nonlinearity']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'ipc' not in engine_params.keys():
                engine_params['ipc'] = GalSimEngine.defaults('Roman')['ipc']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'read_noise' not in engine_params.keys():
                engine_params['read_noise'] = GalSimEngine.defaults('Roman')['read_noise']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            return engine_params
        elif instrument_name.lower() == 'hwo':
            if 'rng_seed' not in engine_params.keys():
                engine_params['rng_seed'] = GalSimEngine.defaults('HWO')['rng_seed']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'sky_background' not in engine_params.keys():
                engine_params['sky_background'] = GalSimEngine.defaults('HWO')['sky_background']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'detector_effects' not in engine_params.keys():
                engine_params['detector_effects'] = GalSimEngine.defaults('HWO')['detector_effects']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'poisson_noise' not in engine_params.keys():
                engine_params['poisson_noise'] = GalSimEngine.defaults('HWO')['poisson_noise']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'dark_noise' not in engine_params.keys():
                engine_params['dark_noise'] = GalSimEngine.defaults('HWO')['dark_noise']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            if 'read_noise' not in engine_params.keys():
                engine_params['read_noise'] = GalSimEngine.defaults('HWO')['read_noise']
                # TODO logging to inform user of default
            else:
                # TODO validate
                pass
            return engine_params
        else:
            Engine.instrument_not_supported(instrument_name)


    @staticmethod
    def get_roman_exposure(synthetic_image, exposure_time, psf=None, engine_params={},
                        verbose=False):
        # set engine params
        if not engine_params:
            engine_params = GalSimEngine.defaults('Roman')
        else:
            engine_params = GalSimEngine.validate_engine_params('Roman', engine_params)

        # get detector and detector position
        detector = synthetic_image.instrument_params['detector']

        # build rng
        rng = galsim.UniformDeviate(engine_params['rng_seed'])

        # import image to GalSim
        image = galsim.Image(array=synthetic_image.image * exposure_time, scale=synthetic_image.pixel_scale, xmin=0, ymin=0, copy=True)

        # add sky background
        if engine_params['sky_background']:
            bkgs = GalSimEngine.get_roman_sky_background(synthetic_image.band, detector, exposure_time,
                                            num_pix=synthetic_image.num_pix)
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
                    image.addNoise(galsim.PoissonNoise(rng))
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
                    image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, total_dark_current)))
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
                    image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
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
            lens_image = GalSimEngine.get_piece(synthetic_image.lens_surface_brightness, synthetic_image.pixel_scale,
                                synthetic_image.pixel_scale, synthetic_image.pixel_scale, exposure_time,
                                psf_interp, gsparams_kwargs)
            source_image = GalSimEngine.get_piece(synthetic_image.source_surface_brightness, synthetic_image.pixel_scale,
                                    synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                                    psf_interp, gsparams_kwargs)
            results = image, lens_image, source_image
        else:
            results = image

        return results, psf, poisson_noise, reciprocity_failure, dark_noise, nonlinearity, ipc, read_noise


    @staticmethod
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


    @staticmethod
    def get_roman_sky_background(bands, sca, exposure_time, num_pix):
        from mejiro.instruments.roman import Roman

        roman = Roman()

        # was only one band provided as a string? or a list of bands?
        if not isinstance(bands, list):
            bands = [bands]

        bkgs = {}
        for band in bands:
            # build Image
            sky_image = galsim.ImageF(num_pix, num_pix)

            # get minimum zodiacal light in this band in counts/pixel/sec
            sky_level = roman.get_min_zodi(band, sca)

            # "For observations at high galactic latitudes, the Zodi intensity is typically ~1.5x the minimum" (https://roman.gsfc.nasa.gov/science/WFI_technical.html)
            sky_level *= 1.5

            # the stray light level is currently set in GalSim to a pessimistic 10% of sky level
            sky_level *= (1. + galsim.roman.stray_light_fraction)

            # get thermal background in this band in counts/pixel/sec
            thermal_bkg = roman.get_thermal_bkg(band, sca)

            # combine the two backgrounds (still counts/pixel/sec)
            sky_image += sky_level
            sky_image += thermal_bkg

            # convert to counts/pixel
            sky_image *= exposure_time

            # # if the image is oversampled, the sky background must be spread out over more pixels
            # sky_image /= oversample ** 2

            bkgs[band] = sky_image

        return bkgs


    @staticmethod
    def get_hwo_exposure(synthetic_image, exposure_time, psf=None, engine_params={}, verbose=False,
                        **kwargs):
        # set engine params
        if not engine_params:
            engine_params = GalSimEngine.defaults('HWO')
        else:
            engine_params = GalSimEngine.validate_engine_params('HWO', engine_params)

        # build rng
        rng = galsim.UniformDeviate(engine_params['rng_seed'])

        # create interpolated image
        total_interp = galsim.InterpolatedImage(galsim.Image(synthetic_image.image, xmin=0, ymin=0),
                                                scale=synthetic_image.pixel_scale,
                                                flux=np.sum(synthetic_image.image) * exposure_time)

        # get PSF
        if psf is None:
            psf_interp = GalSimEngine.get_gaussian_psf(synthetic_image.instrument.get_psf_fwhm(synthetic_image.band))
            psf = psf_interp.drawImage(nx=synthetic_image.native_num_pix, ny=synthetic_image.native_num_pix,
                                    scale=synthetic_image.native_pixel_scale)

        # convolve with PSF
        gsparams_kwargs = engine_params['gsparams_kwargs']
        convolved = galsim.Convolve(total_interp, psf_interp, gsparams=galsim.GSParams(**gsparams_kwargs))

        # draw image at the native pixel scale
        im = galsim.ImageF(synthetic_image.native_num_pix, synthetic_image.native_num_pix,
                        scale=synthetic_image.native_pixel_scale)
        im.setOrigin(0, 0)
        image = convolved.drawImage(im)

        # NB from here on out, the image is at the native pixel scale, i.e., the image is NOT oversampled

        # add sky background
        if engine_params['sky_background']:
            min_zodi_cps = 0.2  # TODO this is a placeholder value
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
                    image.addNoise(galsim.PoissonNoise(rng))
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
                    image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, total_dark_current)))
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
                    image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
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
            lens_image = GalSimEngine.get_piece(synthetic_image.lens_surface_brightness, synthetic_image.pixel_scale,
                                synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                                psf_interp, gsparams_kwargs)
            source_image = GalSimEngine.get_piece(synthetic_image.source_surface_brightness, synthetic_image.pixel_scale,
                                    synthetic_image.native_pixel_scale, synthetic_image.native_num_pix, exposure_time,
                                    psf_interp, gsparams_kwargs)
            results = image, lens_image, source_image
        else:
            results = image

        return results, psf, poisson_noise, dark_noise, read_noise


    @staticmethod
    def get_gaussian_psf(fwhm, flux=1.):
        """
        Generate a Gaussian Point Spread Function (PSF) using GalSim.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum.
        flux : float, optional
            Transmission. Default is 1.

        Returns
        -------
        galsim.Gaussian
            A GalSim Gaussian object representing the PSF.
        """
        return galsim.Gaussian(fwhm=fwhm, flux=flux)


    @staticmethod
    def get_roman_psf(band, detector=1, detector_position=(2048, 2048), pupil_bin=1):
        """
        Generate a Point Spread Function (PSF) for the Roman Space Telescope using GalSim. Note that mejiro's preferred method of generating Roman PSFs is using `STPSF`.

        Parameters
        ----------
        band : str
            The band for which the PSF is to be generated.
        detector : int, optional
            The Roman detector number (SCA), by default 1.
        detector_position : tuple of float, optional
            The position on the detector in pixels, by default (2048, 2048).
        pupil_bin : int, optional
            The binning factor for the pupil plane, by default 1.

        Returns
        -------
        galsim.GSObject
            The PSF object for the specified parameters.
        """
        return galsim.roman.getPSF(SCA=detector,
                                SCA_pos=galsim.PositionD(*detector_position),
                                bandpass=None,
                                wavelength=roman_util.translate_band(band),
                                pupil_bin=pupil_bin)


    @staticmethod
    def get_empty_image(num_pix, pixel_scale):
        """
        Create an empty image using GalSim.

        Parameters
        ----------
        num_pix : int
            The number of pixels along each dimension of the image.
        pixel_scale : float
            The scale of each pixel in the image.

        Returns
        -------
        galsim.ImageD
            An empty image with the specified dimensions and pixel scale with dtype float64.
        """
        return galsim.ImageD(num_pix, num_pix, scale=pixel_scale)
