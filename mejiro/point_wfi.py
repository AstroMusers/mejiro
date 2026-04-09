import copy
import logging
import math
import os
from collections import namedtuple
from typing import List, Tuple, Union

import numpy as np
import pysiaf

import mejiro

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'roman_dither_patterns')
_GAP_FILE = os.path.join(_DATA_DIR, 'WfiImagingGap.txt')
_SUBPIXEL_FILE = os.path.join(_DATA_DIR, 'WfiImagingSubpixel.txt')

SourcePlacement = namedtuple('SourcePlacement', ['sky_coord', 'pixel_offset'])
PixelPosition = namedtuple('PixelPosition', ['cx', 'cy', 'on_detector'])


def _parse_dither_file(filepath):
    """Parse a dither pattern text file into a dict of pattern name -> offsets.

    Parameters
    ----------
    filepath : str
        Path to the dither pattern text file.

    Returns
    -------
    dict
        Mapping of pattern name to list of (x_offset, y_offset) tuples in
        arcseconds.
    """
    patterns = {}
    current_name = None
    current_offsets = []

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if current_name is not None:
                    patterns[current_name] = current_offsets
                    current_name = None
                    current_offsets = []
                continue

            tokens = stripped.split()
            # step lines have 3 tokens: step_number  x_offset  y_offset
            if len(tokens) == 3 and tokens[0].isdigit():
                x = float(tokens[1])
                y = float(tokens[2])
                current_offsets.append((x, y))
            else:
                # pattern name line
                if current_name is not None:
                    patterns[current_name] = current_offsets
                current_name = stripped
                current_offsets = []

    # handle last pattern if file doesn't end with blank line
    if current_name is not None:
        patterns[current_name] = current_offsets

    return patterns


def _make_wcs(pointing, sca, band, ma_table_number, read_pattern, date, pa_aper):
    """Build a romanisim/galsim WCS for a single pointing.

    Parameters
    ----------
    pointing : astropy.coordinates.SkyCoord
        Boresight sky position.
    sca : int
        SCA number (1-18).
    band : str
        Filter name (e.g. 'F158').
    ma_table_number : int
        Multi-accumulation table number.
    read_pattern : list
        Read pattern from romanisim parameters.
    date : str
        Observation date string.
    pa_aper : float
        Position angle of the aperture in degrees.

    Returns
    -------
    galsim.GSFitsWCS
        The WCS for this pointing.
    """
    from astropy.time import Time
    from romanisim import parameters, wcs as romanisim_wcs

    meta = copy.deepcopy(parameters.default_parameters_dictionary)
    meta['instrument']['detector'] = f'WFI{sca:02d}'
    meta['instrument']['optical_element'] = band
    meta['exposure']['ma_table_number'] = ma_table_number
    meta['exposure']['read_pattern'] = read_pattern
    meta['exposure']['start_time'] = Time(date)
    romanisim_wcs.fill_in_parameters(meta, pointing, boresight=True, pa_aper=pa_aper)
    return romanisim_wcs.get_wcs(meta, usecrds=False)


class PointWFI:
    """Represents a Roman WFI pointing with dither pattern support.

    Parameters
    ----------
    ra : float
        Right ascension of the target placed at the geometric center of
        the Wide Field Instrument (WFI) focal plane array, in degrees.
    dec : float
        Declination of the target placed at the geometric center of the
        WFI focal plane array, in degrees.
    position_angle : float
        Position angle of the WFI relative to the V3 axis measured from
        North to East, in degrees. A value of 0.0 degrees would place the
        WFI in the "smiley face" orientation (U-shaped) on the celestial
        sphere. To place WFI such that the position angle of the V3 axis
        is zero degrees, use a WFI position angle of -60 degrees.
    """

    _pattern_cache = None

    def __init__(self, ra=0.0, dec=0.0, position_angle=-60.0):
        self.ra = ra
        self.dec = dec
        self.position_angle = position_angle

        self.siaf_aperture = pysiaf.Siaf('Roman')['WFI_CEN']
        self.v2_ref = self.siaf_aperture.V2Ref
        self.v3_ref = self.siaf_aperture.V3Ref
        self.attitude_matrix = pysiaf.utils.rotations.attitude(
            self.v2_ref, self.v3_ref, self.ra, self.dec, self.position_angle
        )
        self.siaf_aperture.set_attitude_matrix(self.attitude_matrix)
        self.tel_roll = pysiaf.utils.rotations.posangle(self.attitude_matrix, 0, 0)
        self.att0 = self.attitude_matrix.copy()
        self.ra0 = self.ra
        self.dec0 = self.dec

    def __repr__(self):
        return f'PointWFI(ra={self.ra}, dec={self.dec}, position_angle={self.position_angle})'

    def dither(self, x_offset, y_offset):
        """Shift the telescope pointing by offsets in the WFI ideal coordinate
        frame.

        Parameters
        ----------
        x_offset : float
            X offset in arcseconds.
        y_offset : float
            Y offset in arcseconds.
        """
        self.ra, self.dec = self.siaf_aperture.idl_to_sky(x_offset, y_offset)

    @classmethod
    def _load_patterns(cls):
        if cls._pattern_cache is not None:
            return
        cls._pattern_cache = {}
        for filepath in [_GAP_FILE, _SUBPIXEL_FILE]:
            parsed = _parse_dither_file(filepath)
            cls._pattern_cache.update(parsed)
        logger.info(f'Loaded {len(cls._pattern_cache)} dither patterns')

    @classmethod
    def available_patterns(cls):
        """Return a sorted list of all available dither pattern names.

        Returns
        -------
        list of str
            Pattern names from WfiImagingGap.txt and WfiImagingSubpixel.txt.
        """
        cls._load_patterns()
        return sorted(cls._pattern_cache.keys())

    @classmethod
    def get_pattern(cls, name):
        """Get the dither offsets for a named pattern.

        Parameters
        ----------
        name : str
            Pattern name (e.g. 'BOXGAP4_1', 'SUB3').

        Returns
        -------
        list of tuple
            List of (x_offset, y_offset) tuples in arcseconds.

        Raises
        ------
        ValueError
            If the pattern name is not found.
        """
        cls._load_patterns()
        if name not in cls._pattern_cache:
            raise ValueError(
                f"Unknown dither pattern '{name}'. "
                f"Available patterns: {sorted(cls._pattern_cache.keys())}"
            )
        return list(cls._pattern_cache[name])


def find_source_position(pointings, sca, band, tile_size, ma_table_number,
                         date, pa_aper=0.0, read_pattern=None, margin=50,
                         detector_size=4088):
    """Find a sky position where a source tile lands on the detector for all
    dithered pointings.

    For gap-filling dither patterns, the source needs to be placed at a sky
    position that falls on the detector in every dither step. This function
    finds the optimal position by projecting the detector center through all
    dither WCSes and centering within the valid intersection region.

    Parameters
    ----------
    pointings : list of astropy.coordinates.SkyCoord
        Dithered boresight sky positions.
    sca : int
        SCA number (1-18).
    band : str
        Filter name (e.g. 'F158').
    tile_size : int
        Source tile size in pixels.
    ma_table_number : int
        Multi-accumulation table number.
    date : str
        Observation date string.
    pa_aper : float, optional
        Position angle of the aperture in degrees. Default is 0.0.
    read_pattern : list, optional
        Read pattern. If None, derived from ma_table_number.
    margin : int, optional
        Extra pixel margin beyond half_tile. Default is 50.
    detector_size : int, optional
        Detector pixel dimension. Default is 4088.

    Returns
    -------
    SourcePlacement
        Named tuple with ``sky_coord`` (galsim.CelestialCoord) and
        ``pixel_offset`` ((opt_dx, opt_dy) tuple).
    """
    import galsim
    from romanisim import parameters

    if read_pattern is None:
        read_pattern = parameters.read_pattern[ma_table_number]

    half_tile = tile_size // 2
    center = detector_size // 2

    wcses = [
        _make_wcs(pt, sca, band, ma_table_number, read_pattern, date, pa_aper)
        for pt in pointings
    ]

    ref_sky = wcses[0].toWorld(galsim.PositionD(center, center))
    offsets = [wcses[i].toImage(ref_sky) for i in range(len(pointings))]

    lo = half_tile + margin
    hi = detector_size - (half_tile + margin)
    opt_dx = ((max(lo - int(round(p.x)) for p in offsets) +
               min(hi - int(round(p.x)) for p in offsets)) // 2)
    opt_dy = ((max(lo - int(round(p.y)) for p in offsets) +
               min(hi - int(round(p.y)) for p in offsets)) // 2)

    sky_coord = wcses[0].toWorld(galsim.PositionD(center + opt_dx, center + opt_dy))

    return SourcePlacement(sky_coord=sky_coord, pixel_offset=(opt_dx, opt_dy))


def compute_pixel_positions(source_sky, pointings, sca, band, tile_size,
                            ma_table_number, date, pa_aper=0.0,
                            read_pattern=None, detector_size=4088):
    """Compute pixel positions for a source in each dithered pointing.

    Parameters
    ----------
    source_sky : galsim.CelestialCoord
        Sky position of the source.
    pointings : list of astropy.coordinates.SkyCoord
        Dithered boresight sky positions.
    sca : int
        SCA number (1-18).
    band : str
        Filter name (e.g. 'F158').
    tile_size : int
        Source tile size in pixels.
    ma_table_number : int
        Multi-accumulation table number.
    date : str
        Observation date string.
    pa_aper : float, optional
        Position angle of the aperture in degrees. Default is 0.0.
    read_pattern : list, optional
        Read pattern. If None, derived from ma_table_number.
    detector_size : int, optional
        Detector pixel dimension. Default is 4088.

    Returns
    -------
    list of PixelPosition
        List of named tuples with ``cx``, ``cy``, ``on_detector`` for each
        dither step.
    """
    from romanisim import parameters

    if read_pattern is None:
        read_pattern = parameters.read_pattern[ma_table_number]

    half_tile = tile_size // 2
    positions = []

    print(f'{"Step":<6} {"Pixel X":<10} {"Pixel Y":<10} {"On detector?":<14}')
    print('-' * 40)

    for i, dithered_coord in enumerate(pointings):
        imwcs = _make_wcs(dithered_coord, sca, band, ma_table_number,
                          read_pattern, date, pa_aper)

        source_pix = imwcs.toImage(source_sky)
        cx, cy = int(round(source_pix.x)), int(round(source_pix.y))

        on_det = (cx - half_tile >= 0 and cx + half_tile + 1 <= detector_size and
                  cy - half_tile >= 0 and cy + half_tile + 1 <= detector_size)

        positions.append(PixelPosition(cx=cx, cy=cy, on_detector=on_det))
        print(f'{i:<6} {cx:<10} {cy:<10} {"YES" if on_det else "NO -- CLIPPED":<14}')

    return positions


def plot_source_placement(pixel_positions, dither_offsets, tile_size,
                          detector_size=4088):
    """Plot source placement on the detector for each dither step.

    Parameters
    ----------
    pixel_positions : list of PixelPosition
        Pixel positions from ``compute_pixel_positions``.
    dither_offsets : list of tuple
        List of (x_offset, y_offset) tuples in arcseconds.
    tile_size : int
        Source tile size in pixels.
    detector_size : int, optional
        Detector pixel dimension. Default is 4088.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    n = len(pixel_positions)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    half_tile = tile_size // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                             constrained_layout=True, squeeze=False)
    fig.suptitle('Source Position on Detector for Each Dither Step', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue

        cx, cy, on_det = pixel_positions[i]
        x_off, y_off = dither_offsets[i]

        ax.set_xlim(-100, detector_size + 100)
        ax.set_ylim(-100, detector_size + 100)
        det_rect = plt.Rectangle((0, 0), detector_size, detector_size,
                                 fill=False, edgecolor='gray', lw=1.5)
        ax.add_patch(det_rect)

        color = 'tab:green' if on_det else 'tab:red'
        tile_rect = plt.Rectangle((cx - half_tile, cy - half_tile),
                                  tile_size, tile_size, fill=True,
                                  facecolor=color, edgecolor='k', alpha=0.6)
        ax.add_patch(tile_rect)
        ax.plot(cx, cy, 'k+', ms=10, mew=2)

        ax.set_aspect('equal')
        ax.set_title(f'Dither {i}: ({x_off:+.1f}", {y_off:+.1f}")\n'
                     f'pixel ({cx}, {cy})', fontsize=10)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

    return fig, axes
