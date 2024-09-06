import json
import math
import os
from copy import deepcopy

import galsim
import pandas as pd

import mejiro
from mejiro.helpers import psf
from mejiro.instruments.instrument_base import InstrumentBase


class Roman(InstrumentBase):

    def __init__(self):
        name = 'Roman'

        super().__init__(
            name
        )

        module_path = os.path.dirname(mejiro.__file__)
        csv_path = os.path.join(module_path, 'data', 'roman_spacecraft_and_instrument_parameters.csv')
        self.df = pd.read_csv(csv_path)

        # load SCA-specific zeropoints
        self.zp_dict = json.load(open(os.path.join(module_path, 'data', 'roman_zeropoint_magnitudes.json')))

        # ---------------------CONSTANTS---------------------
        self.pixel_scale = 0.11  # arcsec/pixel
        self.diameter = 2.4  # m
        self.psf_jitter = 0.012  # arcsec per axis
        self.pixels_per_axis = 4088
        self.total_pixels_per_axis = 4096
        self.thermal_backgrounds = {
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

    def get_exposure(self, synthetic_image, interp, rng, exposure_time, sky_background=True, detector_effects=True,
                     **kwargs):
        suppress_output = kwargs['suppress_output'] if 'suppress_output' in kwargs else True

        # get PSF
        detector = kwargs['sca']
        detector_position = kwargs['sca_position']
        self.psf = psf.get_webbpsf_psf(synthetic_image.band, detector, detector_position, synthetic_image.oversample,
                                       check_cache=True, psf_cache_dir='/data/bwedig/mejiro/cached_psfs',
                                       suppress_output=suppress_output)
        # self.psf_image = self.psf.drawImage(scale=synthetic_image.pixel_scale)  # TODO FIX

        # convolve with PSF
        convolved = galsim.Convolve([interp, self.psf])

        # draw image
        output_num_pix = math.floor(synthetic_image.num_pix / synthetic_image.oversample)
        im = galsim.ImageF(output_num_pix, output_num_pix, scale=synthetic_image.native_pixel_scale)
        im.setOrigin(0, 0)
        image = convolved.drawImage(im)

        # add sky background
        if sky_background:
            bkgs = self.get_sky_bkgs(synthetic_image.band, exposure_time, num_pix=output_num_pix,
                                     oversample=synthetic_image.native_pixel_scale)
            bkg = bkgs[synthetic_image.band]
            image += bkg

        # add detector effects
        if detector_effects:
            image.replaceNegative(0.)

            # Poisson noise
            if 'poisson_noise' in kwargs:
                image += kwargs['poisson_noise']
            else:
                before = deepcopy(image)
                image.addNoise(galsim.PoissonNoise(rng))
                poisson_noise = image - before
            image.quantize()

            # reciprocity failure
            galsim.roman.addReciprocityFailure(image, exptime=exposure_time)

            # dark current
            if 'dark_noise' in kwargs:
                image += kwargs['dark_noise']
            else:
                before = deepcopy(image)
                total_dark_current = galsim.roman.dark_current
                image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(rng, total_dark_current)))
                dark_noise = image - before

            # skip persistence

            # nonlinearity
            galsim.roman.applyNonlinearity(image)

            # IPC
            galsim.roman.applyIPC(image)

            # read noise
            if 'read_noise' in kwargs:
                image += kwargs['read_noise']
            else:
                before = deepcopy(image)
                read_noise_sigma = galsim.roman.read_noise
                image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_sigma))
                read_noise = image - before

            # gain
            image /= galsim.roman.gain

            # quantize
            image.quantize()

        if detector_effects and 'return_noise' in kwargs and kwargs['return_noise']:
            return image, poisson_noise, dark_noise, read_noise
        else:
            return image

    def get_sky_bkgs(self, bands, exposure_time, num_pix, oversample):
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
            sky_level = self.get_min_zodi(band)

            # "For observations at high galactic latitudes, the Zodi intensity is typically ~1.5x the minimum" (https://roman.gsfc.nasa.gov/science/WFI_technical.html)
            sky_level *= 1.5

            # the stray light level is currently set in GalSim to a pessimistic 10% of sky level
            sky_level *= (1. + galsim.roman.stray_light_fraction)

            # get thermal background in this band in counts/pixel/sec
            thermal_bkg = self.get_thermal_bkg(band)

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
        roman_filters = ['F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F213', 'F146']
        fields = [f'WFI_Filter_{filter}_Center' for filter in roman_filters]

        dict = {}
        for roman_filter, field in zip(roman_filters, fields):
            dict[roman_filter] = float(self.df.loc[self.df['Name'] == field]['Value'].to_string(index=False))

        return dict

    def get_pixel_scale(self):
        return float(self.df.loc[self.df['Name'] == 'WFI_Pixel_Scale']['Value'].to_string(index=False))

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

    def get_thermal_bkg(self, band):
        """
        Return internal thermal background in given band in counts/sec/pixel
        """
        return self.thermal_backgrounds[band.upper()]

    def get_min_zodi(self, band):
        """
        Return minimum zodiacal light level in given band in counts/pixel/sec (count rate per pixel)
        """
        return self.min_zodi[band.upper()]

    def get_zeropoint_magnitude(self, band, sca=1):
        """
        Return AB zeropoint in given band for the given SCA
        """
        sca = Roman.get_sca_string(sca)
        return self.zp_dict[sca][band.upper()]

    def divide_up_sca(self, sides):
        assert sides % 2 == 0, "For now, sides must be even"

        num_centers = sides ** 2
        piece = int(self.pixels_per_axis / num_centers)

        centers = []
        for i in range(num_centers):
            for j in range(num_centers):
                if i % 2 != 0 and j % 2 != 0:
                    centers.append((piece * i, piece * j))
        return centers

    @staticmethod
    def get_sca_string(sca):
        if type(sca) is int:
            return f'SCA{str(sca).zfill(2)}'
        else:
            sca_id = str()

        # might provide int 1 or string 01 or string SCA01
        if type(sca_id) is int:
            return f'SCA{str(sca_id).zfill(2)}'
        else:
            if sca.startswith('SCA'):
                return sca
            else:
                sca = f'SCA{sca_id}'

    @staticmethod
    def get_sca_int(sca):
        if type(sca) is int:
            return sca
        else:
            if sca.startswith('SCA'):
                return int(sca[3:])
            else:
                return int(sca)

    # TODO consider making all methods which might call this one methods on the class that only call this method if a flag on the class (e.g., override) is False. this way, scripts can instantiate Roman() with override=True and then avoid running this method every single time
    @staticmethod
    def translate_band(input):
        # some folks are referring to Roman filters using the corresponding ground-based filters (r, z, Y, etc.)
        options_dict = {
            'F062': ['F062', 'R', 'R062'],
            'F087': ['F087', 'Z', 'Z087'],
            'F106': ['F106', 'Y', 'Y106'],
            'F129': ['F129', 'J', 'J129'],
            'F158': ['F158', 'H', 'H158'],
            'F184': ['F184', 'H/K'],
            'F146': ['F146', 'WIDE', 'W146'],
            'F213': ['F213', 'KS', 'K213']
        }

        # check if input is already a valid band
        if input in options_dict.keys():
            return input

        # capitalize and remove spaces
        input = input.upper().replace(' ', '')

        for band, possible_names in options_dict.items():
            if input in possible_names:
                return band
            else:
                raise ValueError(f"Band {input} not recognized. Valid bands (and aliases) are {options_dict}.")
