"""The L3 registration property, checked against the real dither WCSes.

``process_batch_l3`` cannot be unit-tested end to end -- it runs romanisim and romancal's
MosaicPipeline, which need CRDS -- but the part that was broken is pure geometry: the tile
grid is laid out on integer dither-0 pixels, so every other dither's true position carries a
fractional residual, and the old code discarded it with ``int(round(...))``.

These tests pin (a) that the residuals are real and large, so discarding them is not a
rounding nicety, and (b) that feeding them to ``bin_to_native`` actually produces distinct
per-dither samplings rather than N copies of one image.

See docs/l3_dither_registration.md.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from romanisim import parameters

from mejiro.pipeline._05_romanisim import (
    _dither_pointings,
    bin_to_native,
    compute_overlap_skygrid,
)

# the rung-1 configuration
MA_TABLE = 10
DATE = '2027-05-01T00:00:00'
TILE_SIZE = 91
BAND = 'F129'
SCA = 1


@pytest.fixture(scope='module')
def sub6_residuals():
    """Per-dither fractional pixel residuals of the SUB6 pattern, for a few tile slots."""
    coord = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg)
    pointings = _dither_pointings(coord, 'SUB6')
    wcses, source_skies = compute_overlap_skygrid(
        pointings, SCA, BAND, MA_TABLE, parameters.read_pattern[MA_TABLE], DATE, TILE_SIZE
    )
    assert len(source_skies) > 0

    residuals = []
    for sky in source_skies[:: max(1, len(source_skies) // 8)][:8]:
        per_dither = []
        for w in wcses:
            p = w.toImage(sky)
            per_dither.append((p.x - round(p.x), p.y - round(p.y)))
        residuals.append(per_dither)
    return np.array(residuals)  # (n_slots, n_dithers, 2)


def test_dither_zero_is_exact(sub6_residuals):
    """The grid is built by projecting integer dither-0 pixels to sky, so dither 0 alone
    lands exactly on a pixel. Any other result means the grid construction changed."""
    np.testing.assert_allclose(sub6_residuals[:, 0, :], 0.0, atol=1e-6)


def test_other_dithers_carry_large_residuals(sub6_residuals):
    """SUB6's commanded offsets are 0.019-0.090 arcsec, i.e. 0.17-0.82 detector px. Rounding
    those away is not a rounding nicety -- it replaces the commanded sub-pixel pattern with
    an uncontrolled residual."""
    spread = sub6_residuals.max(axis=1) - sub6_residuals.min(axis=1)  # (n_slots, 2)
    assert spread.max() > 0.5
    # and no tile slot escapes it by having every dither round the same way
    assert (spread.max(axis=1) > 0.1).all()


def test_phases_produce_distinct_detector_samplings(sub6_residuals):
    """The payoff: binning one oversampled tile at each dither's phase gives six different
    detector realizations. The old path placed six identical copies, which is what made the
    co-add an average of sub-pixel-shifted duplicates."""
    ov = 5
    n = 21
    yy, xx = np.mgrid[:n * ov, :n * ov] - (n * ov - 1) / 2.0
    tile = np.exp(-(xx ** 2 + yy ** 2) / (2 * 6.0 ** 2))
    tile /= tile.sum()

    phases = sub6_residuals[0]
    binned = [bin_to_native(tile, ov, fx, fy) for fx, fy in phases]

    # every dither carries the same total ...
    for b in binned:
        assert b.sum() == pytest.approx(tile.sum(), rel=1e-12)

    # ... but distributes it differently across the detector pixels
    for i in range(1, len(binned)):
        assert not np.allclose(binned[i], binned[0]), f'dither {i} sampled identically to dither 0'

    peaks = np.array([b.max() for b in binned])
    assert np.ptp(peaks) / peaks.mean() > 0.01
