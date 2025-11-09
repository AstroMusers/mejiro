import os
import galsim
import galsim.roman  # NB not automatically imported with `import galsim`
import numpy as np
import warnings
from copy import deepcopy

from mejiro.engines.engine import Engine
from mejiro.utils import roman_util


class GalSimEngine(Engine):

    @staticmethod
    def get_roman_exposure(synthetic_image, exposure_time, engine_params={}, verbose=False):
        from mejiro.instruments.roman import Roman
        roman = Roman()

        # set engine params
        if not engine_params:
            engine_params = GalSimEngine.defaults('Roman')
        else:
            engine_params = GalSimEngine.validate_engine_params('Roman', engine_params)

        # build rng
        if 'rng_seed' in engine_params:
            rng = galsim.UniformDeviate(engine_params['rng_seed'])
        else:
            rng = galsim.UniformDeviate()

        # import image to GalSim
        image = galsim.Image(array=synthetic_image.image * exposure_time, scale=synthetic_image.pixel_scale, xmin=0, ymin=0, copy=True)

        # add sky background
        if type(engine_params['sky_background']) is galsim.Image:
            sky_background = engine_params['sky_background']
            image += sky_background
        elif type(engine_params['sky_background']) is bool:
            if engine_params['sky_background']:
                sky_background = GalSimEngine.get_roman_sky_background(roman, synthetic_image.band, exposure_time, synthetic_image.num_pix, engine_params['min_zodi_factor'])
                image += sky_background
            else:
                sky_background = None

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
                    total_dark_current = galsim.roman.dark_current * exposure_time  # TODO would like to get this from roman-technical-information through Roman() instead of galsim
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
                    read_noise_sigma = galsim.roman.read_noise  # TODO would like to get this from roman-technical-information through Roman() instead of galsim
                    image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
                    read_noise = image - before
                else:
                    read_noise = None

            # gain
            image /= galsim.roman.gain

            # quantize, since analog-to-digital conversion gives integers
            image.quantize()
        else:
            poisson_noise = None
            reciprocity_failure = None
            dark_noise = None
            nonlinearity = None
            ipc = None
            read_noise = None

        if synthetic_image.pieces:
            lens_image = GalSimEngine.get_piece(synthetic_image.lens_surface_brightness, exposure_time, synthetic_image.pixel_scale)
            source_image = GalSimEngine.get_piece(synthetic_image.source_surface_brightness, exposure_time, synthetic_image.pixel_scale)
            results = image, lens_image, source_image
        else:
            results = image

        return results, sky_background, poisson_noise, reciprocity_failure, dark_noise, nonlinearity, ipc, read_noise


    @staticmethod
    def get_piece(array, exposure_time, pixel_scale):
        return galsim.ImageD(array=array * exposure_time, 
                            scale=pixel_scale, 
                            xmin=0, 
                            ymin=0, 
                            copy=True)


    @staticmethod
    def get_roman_sky_background(roman, band, exposure_time, num_pix, min_zodi_factor=1.5):
        # build Image
        sky_image = galsim.ImageD(num_pix, num_pix)

        # get minimum zodiacal light in this band in counts/pixel/sec
        sky_level = roman.get_minimum_zodiacal_light(band)
        if sky_level.unit != 'ct / pix':
            raise ValueError(f"Minimum zodiacal light is not in units of counts/pixel: {sky_level.unit}")

        # "For observations at high galactic latitudes, the Zodi intensity is typically ~1.5x the minimum" (https://roman.gsfc.nasa.gov/science/WFI_technical.html)
        sky_level *= min_zodi_factor

        # add stray light contribution
        sky_level *= (1. + roman.stray_light_fraction)

        # get thermal background in this band in counts/pixel/sec
        thermal_bkg = roman.get_thermal_background(band)
        if thermal_bkg.unit != 'ct / pix':
            raise ValueError(f"Thermal background is not in units of counts/pixel: {thermal_bkg.unit}")

        # combine the two backgrounds (still counts/pixel/sec)
        sky_image += sky_level.value
        sky_image += thermal_bkg.value

        # convert to counts/pixel
        sky_image *= exposure_time

        return sky_image
    


    @staticmethod
    def get_sky_background(instrument, band, exposure_time, num_pix, pixel_scale):
        # build Image
        sky_image = GalSimEngine.get_empty_image(num_pix, pixel_scale)

        # get minimum zodiacal light in this band in counts/pixel/sec
        sky_level = instrument.get_sky_level(band)
        if sky_level.unit != 'ct / pix':
            raise ValueError(f"Minimum zodiacal light is not in units of counts/pixel: {sky_level.unit}")

        # add stray light contribution
        sky_level *= (1. + instrument.stray_light_fraction)

        # get thermal background in this band in counts/pixel/sec
        thermal_bkg = instrument.get_thermal_background(band)
        if thermal_bkg.unit != 'ct / pix':
            raise ValueError(f"Thermal background is not in units of counts/pixel: {thermal_bkg.unit}")

        # combine the two backgrounds (still counts/pixel/sec)
        sky_image += sky_level.value
        sky_image += thermal_bkg.value

        # convert to counts/pixel
        sky_image *= exposure_time

        return sky_image


    @staticmethod
    def get_exposure(synthetic_image, exposure_time, engine_params={}, verbose=False):
        # import instrument dynamically based on instrument name
        import importlib
        module = importlib.import_module(f'mejiro.instruments.{synthetic_image.instrument_name.lower()}')
        instrument = getattr(module, synthetic_image.instrument_name)()

        # set engine params
        if not engine_params:
            engine_params = GalSimEngine.defaults(instrument.name)
        else:
            engine_params = GalSimEngine.validate_engine_params(instrument.name, engine_params)

        # build rng
        if 'rng_seed' in engine_params:
            rng = galsim.UniformDeviate(engine_params['rng_seed'])
        else:
            rng = galsim.UniformDeviate()

        # import image to GalSim
        image = galsim.Image(array=synthetic_image.image * exposure_time, scale=synthetic_image.pixel_scale, xmin=0, ymin=0, copy=True)

        # add sky background
        if type(engine_params['sky_background']) is galsim.Image:
            sky_background = engine_params['sky_background']
            image += sky_background
        elif type(engine_params['sky_background']) is bool:
            if engine_params['sky_background']:
                sky_background = GalSimEngine.get_sky_background(instrument, synthetic_image.band, exposure_time, synthetic_image.num_pix, synthetic_image.pixel_scale)
                image += sky_background
            else:
                sky_background = None
        
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
                    total_dark_current = instrument.get_dark_current(synthetic_image.band).value * exposure_time
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
                    read_noise_sigma = instrument.get_read_noise(synthetic_image.band).value
                    image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
                    read_noise = image - before
                else:
                    read_noise = None

            # gain
            image /= instrument.get_gain(synthetic_image.band)

            # quantize, since analog-to-digital conversion gives integers
            image.quantize()
        else:
            poisson_noise = None
            dark_noise = None
            read_noise = None

        if synthetic_image.pieces:
            lens_image = GalSimEngine.get_piece(synthetic_image.lens_surface_brightness, exposure_time, synthetic_image.pixel_scale)
            source_image = GalSimEngine.get_piece(synthetic_image.source_surface_brightness, exposure_time, synthetic_image.pixel_scale)
            results = image, lens_image, source_image
        else:
            results = image

        return results, sky_background, poisson_noise, dark_noise, read_noise


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
        Generate a Point Spread Function (PSF) for the Roman Space Telescope using GalSim. Note that mejiro's preferred package for generating Roman PSFs is `STPSF`.

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
        Create an empty image using GalSim with dtype np.float64.

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
    
    
    @staticmethod
    def defaults(instrument_name):
        if instrument_name.lower() == 'roman':
            return {
                # 'rng_seed': 42,
                'min_zodi_factor': 1.5,
                'sky_background': True,
                'detector_effects': True,
                'poisson_noise': True,
                'reciprocity_failure': True,
                'dark_noise': True,
                'nonlinearity': True,
                'ipc': True,
                'read_noise': True,
            }
        elif instrument_name.lower() == 'hwo' or instrument_name.lower() == 'jwst' or instrument_name.lower() == 'hst':
            return {
                # 'rng_seed': 42,
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
            # if 'rng_seed' not in engine_params.keys():
            #     engine_params['rng_seed'] = GalSimEngine.defaults('Roman')['rng_seed']
                # TODO logging to inform user of default
            # else:
            #     # TODO validate
            #     pass
            if 'min_zodi_factor' not in engine_params.keys():
                engine_params['min_zodi_factor'] = GalSimEngine.defaults('Roman')['min_zodi_factor']
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
        elif instrument_name.lower() == 'hwo' or instrument_name.lower() == 'jwst' or instrument_name.lower() == 'hst':
            # if 'rng_seed' not in engine_params.keys():
            #     engine_params['rng_seed'] = GalSimEngine.defaults('HWO')['rng_seed']
                # TODO logging to inform user of default
            # else:
            #     # TODO validate
            #     pass
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
