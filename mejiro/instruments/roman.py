import json
import math
import os
import warnings
from copy import deepcopy

import galsim
import pandas as pd

import mejiro
from mejiro.instruments.instrument_base import InstrumentBase
from mejiro.utils import roman_util


class Roman(InstrumentBase):

    def __init__(self):
        name = 'Roman'
        bands = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        engines = ['galsim', 'pandeia', 'romanisim']

        super().__init__(
            name,
            bands,
            engines
        )

        module_path = os.path.dirname(mejiro.__file__)
        csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
        self.df = pd.read_csv(csv_path)

        # load SCA-specific zeropoints
        self.zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))
        self.min_zodi_dict = json.load(open(os.path.join(module_path, 'data', 'roman_minimum_zodiacal_light.json')))
        self.thermal_bkg_dict = json.load(open(os.path.join(module_path, 'data', 'roman_thermal_background.json')))

        # ---------------------CONSTANTS---------------------
        self.pixel_scale = 0.11  # arcsec per pixel
        self.diameter = 2.4  # m
        self.psf_jitter = 0.012  # arcsec per axis
        self.pixels_per_axis = 4088
        self.total_pixels_per_axis = 4096
        self.thermal_bkg = {
            'F062': 0.003,
            'F087': 0.003,
            'F106': 0.003,
            'F129': 0.003,
            'F158': 0.048,
            'F184': 0.155,
            'F213': 4.38,
            'F146': 1.03
        }  # retrieved 31 July 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html
        self.min_zodi = {
            'F062': 0.25,
            'F087': 0.251,
            'F106': 0.277,
            'F129': 0.267,
            'F158': 0.244,
            'F184': 0.141,
            'F213': 0.118,
            'F146': 0.781
        }  # retrieved 31 July 2024 from https://roman.gsfc.nasa.gov/science/WFI_technical.html
        self.psf_fwhm = {
            'F062': 0.058,
            'F087': 0.073,
            'F106': 0.087,
            'F129': 0.106,
            'F158': 0.128,
            'F184': 0.146,
            'F213': 0.169,
            'F146': 0.105
        }  # retrieved 25 June 2024 from https://outerspace.stsci.edu/pages/viewpage.action?spaceKey=ISWG&title=Roman+WFI+and+Observatory+Performance

    def validate_instrument_config(config):
        # TODO implement this
        pass

    def get_pixel_scale(self, band=None):
        """
        Returns the pixel scale for Roman's WFI.

        Parameters:
        band (optional): The specific band for which to get the pixel scale. For Roman's WFI, the pixel scale is the same across all bands.

        Returns:
        float: The pixel scale in arcseconds per pixel.
        """
        return self.pixel_scale

    def get_exposure(self, synthetic_image, interp, rng, exposure_time, sky_background=True, detector_effects=True,
                     **kwargs):
        suppress_output = kwargs['suppress_output'] if 'suppress_output' in kwargs else True

        # get PSF
        detector = kwargs['sca']
        detector_position = kwargs['sca_position']
        from mejiro.helpers import psf
        self.psf = psf.get_webbpsf_psf(band=synthetic_image.band, 
                                       detector=detector, 
                                       detector_position=detector_position, 
                                       oversample=synthetic_image.oversample,
                                       num_pix=synthetic_image.native_num_pix,  # NB WebbPSF wants the native pixel size
                                       check_cache=True, 
                                       psf_cache_dir='/data/bwedig/mejiro/cached_psfs',
                                       suppress_output=suppress_output)
        # self.psf_image = self.psf.drawImage(scale=synthetic_image.pixel_scale)  # TODO FIX

        # convolve with PSF
        convolved = galsim.Convolve(interp, self.psf)

        # draw image at the native pixel scale
        im = galsim.ImageF(synthetic_image.native_num_pix, synthetic_image.native_num_pix, scale=synthetic_image.native_pixel_scale)
        im.setOrigin(0, 0)
        image = convolved.drawImage(im)

        # NB from here on out, the image is at the native pixel scale, i.e., the image is NOT oversampled

        # add sky background
        if sky_background:
            bkgs = self.get_sky_bkgs(synthetic_image.band, detector, exposure_time, num_pix=synthetic_image.native_num_pix, oversample=1)
            bkg = bkgs[synthetic_image.band]
            image += bkg
        
        # integer number of photons are being detected, so quantize
        image.quantize()

        # add detector effects
        if detector_effects:
            image.replaceNegative(0.)

            # Poisson noise
            if 'poisson_noise' not in kwargs:
                before = deepcopy(image)
                image.addNoise(galsim.PoissonNoise(rng))
                poisson_noise = image - before
                image.quantize()
            elif type(kwargs['poisson_noise']) is galsim.Image:
                image += kwargs['poisson_noise']
                image.quantize()
            elif type(kwargs['poisson_noise']) is bool:
                poisson_noise = None
                pass
            
            # reciprocity failure
            if 'reciprocity_failure' not in kwargs:
                before = deepcopy(image)
                galsim.roman.addReciprocityFailure(image, exptime=exposure_time)
                reciprocity_failure = image - before
            elif type(kwargs['reciprocity_failure']) is galsim.Image:
                image += kwargs['reciprocity_failure']
            elif type(kwargs['reciprocity_failure']) is bool:
                reciprocity_failure = None
                pass                

            # dark current
            if 'dark_noise' not in kwargs:
                before = deepcopy(image)
                total_dark_current = galsim.roman.dark_current
                image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, total_dark_current)))
                dark_noise = image - before
            elif type(kwargs['dark_noise']) is galsim.Image:
                image += kwargs['dark_noise']
            elif type(kwargs['dark_noise']) is bool:
                dark_noise = None
                pass

            # skip persistence

            # nonlinearity
            if 'nonlinearity' not in kwargs:
                before = deepcopy(image)
                galsim.roman.applyNonlinearity(image)
                nonlinearity = image - before
            elif type(kwargs['nonlinearity']) is galsim.Image:
                image += kwargs['nonlinearity']
            elif type(kwargs['nonlinearity']) is bool:
                nonlinearity = None
                pass

            # IPC
            if 'ipc' not in kwargs:
                before = deepcopy(image)
                galsim.roman.applyIPC(image)
                ipc = image - before
            elif type(kwargs['ipc']) is galsim.Image:
                image += kwargs['ipc']
            elif type(kwargs['ipc']) is bool:
                ipc = None
                pass

            # read noise
            if 'read_noise' not in kwargs:
                before = deepcopy(image)
                read_noise_sigma = galsim.roman.read_noise
                image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
                read_noise = image - before
            elif type(kwargs['read_noise']) is galsim.Image:
                image += kwargs['read_noise']
            elif type(kwargs['read_noise']) is bool:
                read_noise = None
                pass

            # gain
            image /= galsim.roman.gain

            # quantize
            image.quantize()

        if detector_effects and 'return_noise' in kwargs and kwargs['return_noise']:
            return image, poisson_noise, reciprocity_failure, dark_noise, nonlinearity, ipc, read_noise
        else:
            return image

    def get_sky_bkgs(self, bands, sca, exposure_time, num_pix, oversample):
        # was only one band provided as a string? or a list of bands?
        single_band = False
        if not isinstance(bands, list):
            single_band = True
            bands = [bands]

        bkgs = {}
        for band in bands:
            # build Image
            sky_image = galsim.ImageF(num_pix, num_pix)

            # get minimum zodiacal light in this band in counts/pixel/sec
            sky_level = self.get_min_zodi(band, sca)

            # "For observations at high galactic latitudes, the Zodi intensity is typically ~1.5x the minimum" (https://roman.gsfc.nasa.gov/science/WFI_technical.html)
            sky_level *= 1.5

            # the stray light level is currently set in GalSim to a pessimistic 10% of sky level
            sky_level *= (1. + galsim.roman.stray_light_fraction)

            # get thermal background in this band in counts/pixel/sec
            thermal_bkg = self.get_thermal_bkg(band, sca)

            # combine the two backgrounds (still counts/pixel/sec)
            sky_image += sky_level
            sky_image += thermal_bkg

            # convert to counts/pixel
            sky_image *= exposure_time

            # if the image is oversampled, the sky background must be spread out over more pixels
            sky_image /= oversample ** 2

            bkgs[band] = sky_image

        return bkgs

    def get_filter_centers(self):
        fields = [f'WFI_Filter_{band}_Center' for band in self.bands]

        filter_centers = {}
        for band, field in zip(self.bands, fields):
            filter_centers[band] = float(self.df.loc[self.df['Name'] == field]['Value'].to_string(index=False))

        return filter_centers

    def get_min_max_wavelength(self, band):
        range = self.df.loc[self.df['Name'] == f'WFI_Filter_{band.upper()}_Wavelength_Range']['Value'].to_string(
            index=False)
        min, max = range.split('-')
        return float(min), float(max)

    def get_min_zodi_count_rate(self, band):
        count_rate = self.df.loc[self.df['Name'] == f'WFI_Count_Rate_Zody_Minimum_{band.upper()}']['Value'].to_string(
            index=False)
        return float(count_rate)

    def get_psf_fwhm(self, band):
        """
        Return PSF FWHM in given band in arcsec. Note from STScI: "PSF FWHM in arcseconds simulated for a detector near the center of the WFI FOV using an input spectrum for a K0V type star."
        """
        return self.psf_fwhm[band.upper()]

    def get_thermal_bkg(self, band, sca):
        sca = roman_util.get_sca_string(sca)
        return self.thermal_bkg_dict[sca][band.upper()]

    def get_min_zodi(self, band, sca):
        sca = roman_util.get_sca_string(sca)
        return self.min_zodi_dict[sca][band.upper()]

    def get_zeropoint_magnitude(self, band, sca=1):
        """
        Return AB zeropoint in given band for the given SCA
        """
        sca = roman_util.get_sca_string(sca)
        return self.zp_dict[sca][band.upper()]

    def divide_up_sca(self, sides):  # TODO move to utils.roman_utils
        sub_array_size = self.pixels_per_axis / sides
        centers = []

        for i in range(sides):
            for j in range(sides):
                center_x = int(round((i + 0.5) * sub_array_size))
                center_y = int(round((j + 0.5) * sub_array_size))
                centers.append((center_x, center_y))
        
        return centers
