import numpy as np
import galsim
from galsim import roman
from romanisim import image, parameters, catalog, psf, util, wcs, persistence
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy import table
import asdf
from astropy.modeling.functional_models import Sersic2D
from romanisim import log
from roman_datamodels.stnode import WfiScienceRaw, WfiImage
import romanisim.bandpass
from astropy.io import fits
import warnings
from copy import deepcopy

from mejiro.engines.engine import Engine
from mejiro.utils import roman_util


class RomanISimEngine(Engine):
    @staticmethod
    def defaults(instrument_name):
        if instrument_name.lower() == 'roman':
            return {

            }
        else:
            Engine.instrument_not_supported(instrument_name)

    @staticmethod
    def validate_engine_params(engine_params):
        # TODO implement
        pass

    @staticmethod
    def get_roman_exposure(synthetic_image, exposure_time, psf=None, engine_params=defaults('Roman'),
                           verbose=False):
        # set engine params
        if not engine_params:
            engine_params = RomanISimEngine.defaults('Roman')
        else:
            engine_params = RomanISimEngine.validate_engine_params('Roman', engine_params)
        