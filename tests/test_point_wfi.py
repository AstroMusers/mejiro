import os
import pytest
import numpy as np

import mejiro
from mejiro.point_wfi import (
    PointWFI,
    SourcePlacement,
    PixelPosition,
    _parse_dither_file,
    find_source_position,
    compute_pixel_positions,
)


_DATA_DIR = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'roman_dither_patterns')
_GAP_FILE = os.path.join(_DATA_DIR, 'WfiImagingGap.txt')
_SUBPIXEL_FILE = os.path.join(_DATA_DIR, 'WfiImagingSubpixel.txt')


# ---------------------------------------------------------------------------
# Dither pattern parsing
# ---------------------------------------------------------------------------

class TestParseDitherFile:

    def test_parse_gap_patterns(self):
        patterns = _parse_dither_file(_GAP_FILE)
        assert len(patterns) == 28
        assert 'BOXGAP4_1' in patterns
        assert 'LINEGAP2_1' in patterns

    def test_parse_subpixel_patterns(self):
        patterns = _parse_dither_file(_SUBPIXEL_FILE)
        assert len(patterns) == 8
        assert 'SUB2' in patterns
        assert 'SUB9' in patterns

    def test_gap_pattern_step_counts(self):
        patterns = _parse_dither_file(_GAP_FILE)
        assert len(patterns['BOXGAP4_1']) == 4
        assert len(patterns['LINEGAP2_1']) == 2
        assert len(patterns['BOXGAP9_2']) == 9

    def test_subpixel_pattern_step_counts(self):
        patterns = _parse_dither_file(_SUBPIXEL_FILE)
        assert len(patterns['SUB2']) == 2
        assert len(patterns['SUB9']) == 9


class TestAvailablePatterns:

    def test_total_count(self):
        patterns = PointWFI.available_patterns()
        assert len(patterns) == 36

    def test_contains_gap_and_subpixel(self):
        patterns = PointWFI.available_patterns()
        assert 'BOXGAP4_1' in patterns
        assert 'LINEGAP5_4' in patterns
        assert 'SUB2' in patterns
        assert 'SUB9' in patterns

    def test_sorted(self):
        patterns = PointWFI.available_patterns()
        assert patterns == sorted(patterns)


class TestGetPattern:

    def test_boxgap4_1(self):
        offsets = PointWFI.get_pattern('BOXGAP4_1')
        assert len(offsets) == 4
        assert offsets[0] == (0.0, -0.0)
        assert offsets[1] == (-205.20, 0.88)
        assert offsets[2] == (-204.32, 206.08)
        assert offsets[3] == (0.88, 205.20)

    def test_sub2(self):
        offsets = PointWFI.get_pattern('SUB2')
        assert len(offsets) == 2
        assert offsets[0] == (0.0, 0.0)
        assert offsets[1] == (0.0275, 0.0825)

    def test_invalid_pattern(self):
        with pytest.raises(ValueError, match="Unknown dither pattern"):
            PointWFI.get_pattern('NONEXISTENT_PATTERN')

    def test_returns_copy(self):
        offsets1 = PointWFI.get_pattern('BOXGAP4_1')
        offsets1.append((999.0, 999.0))
        offsets2 = PointWFI.get_pattern('BOXGAP4_1')
        assert len(offsets2) == 4


# ---------------------------------------------------------------------------
# PointWFI class
# ---------------------------------------------------------------------------

class TestPointWFI:

    def test_init_defaults(self):
        p = PointWFI()
        assert p.ra == 0.0
        assert p.dec == 0.0
        assert p.position_angle == -60.0
        assert p.siaf_aperture is not None
        assert p.attitude_matrix is not None

    def test_init_custom(self):
        p = PointWFI(ra=45.0, dec=-30.0, position_angle=0.0)
        assert p.ra == 45.0
        assert p.dec == -30.0
        assert p.position_angle == 0.0

    def test_originals_saved(self):
        p = PointWFI(ra=10.0, dec=20.0)
        assert p.ra0 == 10.0
        assert p.dec0 == 20.0
        assert p.att0 is not None

    def test_dither_zero(self):
        p = PointWFI(ra=45.0, dec=-30.0, position_angle=0.0)
        p.dither(0.0, 0.0)
        assert abs(p.ra - 45.0) < 0.01
        assert abs(p.dec - (-30.0)) < 0.01

    def test_dither_nonzero(self):
        p = PointWFI(ra=45.0, dec=-30.0, position_angle=0.0)
        original_ra = p.ra
        original_dec = p.dec
        p.dither(-205.20, 0.88)
        assert p.ra != original_ra or p.dec != original_dec

    def test_repr(self):
        p = PointWFI(ra=1.0, dec=2.0, position_angle=3.0)
        r = repr(p)
        assert 'PointWFI' in r
        assert '1.0' in r
        assert '2.0' in r
        assert '3.0' in r


# ---------------------------------------------------------------------------
# find_source_position and compute_pixel_positions (require romanisim)
# ---------------------------------------------------------------------------

try:
    import romanisim
    HAS_ROMANISIM = True
except ImportError:
    HAS_ROMANISIM = False

try:
    import galsim
    HAS_GALSIM = True
except ImportError:
    HAS_GALSIM = False


@pytest.mark.skipif(not HAS_ROMANISIM, reason='romanisim not installed')
@pytest.mark.skipif(not HAS_GALSIM, reason='galsim not installed')
class TestFindSourcePosition:

    @pytest.fixture
    def pointings(self):
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        offsets = PointWFI.get_pattern('BOXGAP4_1')
        pts = []
        for x_off, y_off in offsets:
            p = PointWFI(ra=0.5, dec=-44.0, position_angle=0.0)
            p.dither(x_off, y_off)
            pts.append(SkyCoord(ra=p.ra * u.deg, dec=p.dec * u.deg))
        return pts

    @pytest.fixture
    def obs_params(self):
        return {
            'sca': 1,
            'band': 'F158',
            'tile_size': 73,
            'ma_table_number': 3,
            'date': '2027-01-01T00:00:00',
            'pa_aper': 0.0,
        }

    def test_returns_source_placement(self, pointings, obs_params):
        result = find_source_position(pointings=pointings, **obs_params)
        assert isinstance(result, SourcePlacement)
        assert hasattr(result.sky_coord, 'ra')
        assert hasattr(result.sky_coord, 'dec')
        assert isinstance(result.pixel_offset, tuple)
        assert len(result.pixel_offset) == 2

    def test_pixel_offset_type(self, pointings, obs_params):
        result = find_source_position(pointings=pointings, **obs_params)
        dx, dy = result.pixel_offset
        assert isinstance(dx, (int, np.integer))
        assert isinstance(dy, (int, np.integer))


@pytest.mark.skipif(not HAS_ROMANISIM, reason='romanisim not installed')
@pytest.mark.skipif(not HAS_GALSIM, reason='galsim not installed')
class TestComputePixelPositions:

    @pytest.fixture
    def setup(self):
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        offsets = PointWFI.get_pattern('BOXGAP4_1')
        pointings = []
        for x_off, y_off in offsets:
            p = PointWFI(ra=0.5, dec=-44.0, position_angle=0.0)
            p.dither(x_off, y_off)
            pointings.append(SkyCoord(ra=p.ra * u.deg, dec=p.dec * u.deg))

        obs_params = {
            'sca': 1,
            'band': 'F158',
            'tile_size': 73,
            'ma_table_number': 3,
            'date': '2027-01-01T00:00:00',
            'pa_aper': 0.0,
        }

        result = find_source_position(pointings=pointings, **obs_params)
        return result, pointings, obs_params

    def test_returns_correct_count(self, setup):
        result, pointings, obs_params = setup
        positions = compute_pixel_positions(
            source_sky=result.sky_coord, pointings=pointings, **obs_params
        )
        assert len(positions) == 4

    def test_returns_pixel_positions(self, setup):
        result, pointings, obs_params = setup
        positions = compute_pixel_positions(
            source_sky=result.sky_coord, pointings=pointings, **obs_params
        )
        for pos in positions:
            assert isinstance(pos, PixelPosition)
            assert isinstance(pos.cx, int)
            assert isinstance(pos.cy, int)
            assert isinstance(pos.on_detector, bool)

    def test_all_on_detector(self, setup):
        result, pointings, obs_params = setup
        positions = compute_pixel_positions(
            source_sky=result.sky_coord, pointings=pointings, **obs_params
        )
        for pos in positions:
            assert pos.on_detector, f'Dither at pixel ({pos.cx}, {pos.cy}) is off detector'
