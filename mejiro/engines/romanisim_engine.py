import logging
from copy import deepcopy
import warnings

import numpy as np
from scipy.ndimage import generic_filter

from astropy import table, units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.functional_models import Sersic2D
from astropy.time import Time
import asdf

import romanisim.bandpass
import romanisim.image as rsim_image
import romanisim.wcs as rsim_wcs
import romanisim.parameters as rsim_params
import romanisim.util as rsim_util

from mejiro.engines.engine import Engine
from mejiro.utils import roman_util

logger = logging.getLogger(__name__)


def poisson_stamp_to_extra_counts(stamp, center_x, center_y, npix=None, rng_seed=None):
    """Poisson-sample an expected-count stamp into a full-frame GalSim image."""
    import galsim

    if npix is None:
        npix = galsim.roman.n_pix
    extra_counts = np.zeros((npix, npix), dtype=np.int32)
    stamp_y_size, stamp_x_size = stamp.shape
    half_y, half_x = stamp_y_size // 2, stamp_x_size // 2
    y_low, x_low = center_y - half_y, center_x - half_x
    y_high, x_high = y_low + stamp_y_size, x_low + stamp_x_size
    stamp_y_low, stamp_x_low = max(0, -y_low), max(0, -x_low)
    stamp_y_high = stamp_y_size - max(0, y_high - npix)
    stamp_x_high = stamp_x_size - max(0, x_high - npix)
    y_low_frame, x_low_frame = max(0, y_low), max(0, x_low)
    y_high_frame, x_high_frame = min(npix, y_high), min(npix, x_high)

    generator = np.random.default_rng(rng_seed)
    clipped_stamp = np.clip(
        stamp[stamp_y_low:stamp_y_high, stamp_x_low:stamp_x_high], 0, None
    )
    realized_counts = generator.poisson(clipped_stamp).astype(np.int32)
    extra_counts[y_low_frame:y_high_frame, x_low_frame:x_high_frame] = realized_counts
    return galsim.ImageI(extra_counts)


def _simulate_lens(
    extra_counts,
    band,
    sca,
    ma_table_number,
    coord,
    date,
    rng_seed=42,
    level=2,
    crparam=dict(),
    catalog=None,
    *,
    use_crds,
    psf_type,
):
    """Run the shared RomanISim setup for an injected lens image."""
    import galsim
    from astropy.table import Table

    meta = deepcopy(rsim_params.default_parameters_dictionary)
    meta["instrument"]["detector"] = f"WFI{sca:02d}"
    meta["instrument"]["optical_element"] = band
    meta["exposure"]["ma_table_number"] = ma_table_number
    meta["exposure"]["read_pattern"] = rsim_params.read_pattern[ma_table_number]
    meta["exposure"]["start_time"] = Time(date, format="isot")
    rsim_wcs.fill_in_parameters(meta, coord, boresight=True)
    if catalog is None:
        catalog = Table(
            {
                "ra": np.array([], dtype="f8"),
                "dec": np.array([], dtype="f8"),
                "type": np.array([], dtype="U3"),
                "n": np.array([], dtype="f4"),
                "half_light_radius": np.array([], dtype="f4"),
                "pa": np.array([], dtype="f4"),
                "ba": np.array([], dtype="f4"),
                band: np.array([], dtype="f4"),
            }
        )
    rng = galsim.UniformDeviate(rng_seed)
    return rsim_image.simulate(
        meta,
        catalog,
        usecrds=use_crds,
        psftype=psf_type,
        level=level,
        rng=rng,
        crparam=crparam,
        extra_counts=extra_counts,
    )


def simulate_lens_no_crds(
    extra_counts,
    band,
    sca,
    ma_table_number,
    coord,
    date,
    rng_seed=42,
    level=2,
    crparam=dict(),
    catalog=None,
):
    """Simulate injected lens counts with RomanISim's built-in reference data."""
    return _simulate_lens(
        extra_counts, band, sca, ma_table_number, coord, date, rng_seed,
        level, crparam, catalog, use_crds=False, psf_type="galsim",
    )


def simulate_lens_with_crds(
    extra_counts,
    band,
    sca,
    ma_table_number,
    coord,
    date,
    rng_seed=42,
    level=2,
    crparam=dict(),
    catalog=None,
):
    """Simulate injected lens counts using CRDS and restore global references."""
    saved_reference_data = dict(rsim_params.reference_data)
    try:
        for key in rsim_params.reference_data:
            rsim_params.reference_data[key] = None
        return _simulate_lens(
            extra_counts, band, sca, ma_table_number, coord, date, rng_seed,
            level, crparam, catalog, use_crds=True, psf_type="epsf",
        )
    finally:
        rsim_params.reference_data.update(saved_reference_data)


class RomanISimEngine(Engine):

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
    def build_extra_counts(synthetic_image, exposure_time, engine_params=defaults('Roman')):
        band = synthetic_image.band
        sca = synthetic_image.instrument_params['detector']

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
        lens_electrons = (synth.data / np.sum(synth.data)) * total_electrons

    @staticmethod
    def tile_extra_counts(synthetic_image_list):
        # TODO check that the length of the list and the size of each image is correct
        pass

        # return tiled_extra_counts



    

    @staticmethod
    def validate_engine_params(engine_params):
        # TODO implement
        pass
