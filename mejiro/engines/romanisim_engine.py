import logging
import numpy as np
import galsim
from galsim import roman

logger = logging.getLogger(__name__)
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
from romanisim.l1 import make_l1
from romanisim.image import make_l2, gather_reference_data, make_asdf
from romanisim import parameters, util, ris_make_utils, wcs
import roman_datamodels
import roman_datamodels.maker_utils as maker_utils
from scipy.ndimage import generic_filter

from mejiro.engines.engine import Engine
from mejiro.utils import roman_util


class RomanISimEngine(Engine):

    @staticmethod
    def build_extra_counts(synthetic_image, exposure_time, psf=None, engine_params=defaults('Roman')):
        band = synthetic_image.band
        sca = engine_params['sca']

        # AB zeropoint flux [electrons/s]
        flux_zeropoint = romanisim.bandpass.get_abflux(band, sca)
        logger.debug(f"AB zeropoint flux for {band}, SCA{sca}: {abflux:.3e} e/s")

        # Exposure time from the MA table
        read_pattern = rsim_params.read_pattern[ma_table_number]
        exptime = rsim_params.read_time * read_pattern[-1][-1]
        logger.debug(f"Total exposure time (MA table {ma_table_number}): {exptime:.1f} s")

        # Total electrons from the lens system
        total_electrons = maggies * abflux * exptime
        logger.debug(f"Lens flux: {maggies:.3e} maggies "
            f"(AB mag {-2.5*np.log10(maggies):.1f})")
        logger.debug(f"Total electrons from lens: {total_electrons:.1f}")

        # Scale the normalized stamp to electrons
        lens_electrons = (synth.image / np.sum(synth.image)) * total_electrons

    @staticmethod
    def tile_extra_counts(synthetic_image_list):
        # TODO check that the length of the list and the size of each image is correct
        pass

        # return tiled_extra_counts



    @staticmethod
    def defaults(instrument_name='roman'):
        if instrument_name.lower() == 'roman':
            return {
                'rng_seed': 42,
                'sca': 1,
                'ma_table_number': 4,
                'date': '2027-01-01T00:00:00',
                'coord': SkyCoord(ra=270.0 * u.deg, dec=66.0 * u.deg)
            }
        else:
            Engine.instrument_not_supported(instrument_name)

    @staticmethod
    def validate_engine_params(engine_params):
        # TODO implement
        pass
