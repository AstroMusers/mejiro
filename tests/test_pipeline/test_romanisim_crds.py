"""``exposure.usecrds`` in ``_05_romanisim``.

The tests without a marker check the pieces that don't need CRDS: the relative-QE scaling,
the exposure-time sidecar, and the L3 cutout flip. The ``crds`` tests talk to CRDS and
download reference files into ``CRDS_PATH`` (a few GB for one detector), so they only run
with ``pytest --run-crds``, e.g. on the Roman Research Nexus, where CRDS is set up.
"""
import os

import galsim
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from romanisim.models import parameters

from mejiro.pipeline._05_romanisim import (
    DETECTOR_SIZE,
    PA_APER,
    _apply_relative_qe,
    _crds_qe_inputs,
    _galsim_pixel_to_radec,
    _romanisim_image_model,
    _sky_parity,
)
from mejiro.pipeline.pipeline_helper import PipelineHelper
from mejiro.point_wfi import _make_wcs, _romanisim_metadata

SCA, BAND, MA_TABLE, DATE = 1, 'F129', 17, '2027-04-15T00:00:00'
COORD = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg)


# ---------------------------------------------------------------------------
# relative QE
# ---------------------------------------------------------------------------

def test_relative_qe_follows_the_flat():
    """With uniform gain and pixel area, the QE is the flat normalized to median 1."""
    flat = np.ones((50, 50))
    flat[10:20, 10:20] = 1.1
    flat[30:40, 30:40] = 0.9
    electrons = np.full((50, 50), 7.0)
    np.testing.assert_allclose(_apply_relative_qe(electrons, flat, 2.0, 0.0121), 7.0 * flat)


def test_relative_qe_divides_out_gain_and_area():
    """A CRDS flat is area * QE / gain (normalized). With a uniform QE, the relative QE is 1
    everywhere however the gain and pixel area vary, so the electrons are unchanged."""
    rng = np.random.default_rng(1)
    gain = rng.uniform(1.8, 2.2, (40, 40))
    area = rng.uniform(0.0118, 0.0124, (40, 40))
    flat = area / gain
    flat /= np.median(flat)
    electrons = rng.uniform(0, 100, (40, 40))
    np.testing.assert_allclose(_apply_relative_qe(electrons, flat, gain, area), electrons)


# ---------------------------------------------------------------------------
# exposure-time sidecar
# ---------------------------------------------------------------------------

def _write_sidecars(step_dir, level, exposure_time=None):
    with open(os.path.join(step_dir, 'exposure_level.txt'), 'w') as f:
        f.write(level)
    if exposure_time is not None:
        with open(os.path.join(step_dir, 'exposure_time.txt'), 'w') as f:
            f.write(repr(exposure_time))


def test_l2_exposure_time_comes_from_the_sidecar(tmp_path):
    _write_sidecars(tmp_path, 'l2', 123.4)
    assert PipelineHelper.romanisim_exposure_metadata(str(tmp_path), MA_TABLE) == (123.4, 'DN/s')


def test_l2_without_sidecar_falls_back_to_builtin_ma_table(tmp_path):
    """L2 runs from before the sidecar was written for L2 used romanisim's built-in tables."""
    _write_sidecars(tmp_path, 'l2')
    read_pattern = parameters.read_pattern[MA_TABLE]
    expected = parameters.read_time * read_pattern[-1][-1]
    assert PipelineHelper.romanisim_exposure_metadata(str(tmp_path), MA_TABLE) == (expected, 'DN/s')


def test_l3_exposure_time_comes_from_the_sidecar(tmp_path):
    _write_sidecars(tmp_path, 'l3', 1176.4)
    assert PipelineHelper.romanisim_exposure_metadata(str(tmp_path), MA_TABLE) == (1176.4, 'MJy/sr')


# ---------------------------------------------------------------------------
# L3 cutout flip
# ---------------------------------------------------------------------------

def _north_up_east_left(ra, dec, scale_deg=0.11 / 3600):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [1000, 1000]
    w.wcs.cdelt = [-scale_deg, scale_deg]
    return lambda x, y: tuple(float(v) for v in w.wcs_pix2world(x, y, 0))


def test_conventional_image_has_negative_parity():
    assert _sky_parity(_north_up_east_left(150.0, 2.0), 1000, 1000) == -1


def test_galsim_detector_wcs_needs_the_flip():
    """The L3 path always flipped cutouts in x for galsim's Roman WCS: deriving the flip
    from the parities must keep doing that."""
    imwcs = _make_wcs(COORD, SCA, BAND, MA_TABLE, parameters.read_pattern[MA_TABLE], DATE, PA_APER)
    detector = _sky_parity(_galsim_pixel_to_radec(imwcs), DETECTOR_SIZE // 2, DETECTOR_SIZE // 2)
    mosaic = _sky_parity(_north_up_east_left(COORD.ra.deg, COORD.dec.deg), 1000, 1000)
    assert detector != mosaic


# ---------------------------------------------------------------------------
# with CRDS (pytest --run-crds)
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_romanisim_parameters():
    """The CRDS path sets every ``parameters.reference_data`` entry to None, sets
    ``parameters.read_time``, and loads an MA table into ``parameters.ma_table_reference``; put
    them back for the tests after."""
    reference_data = dict(parameters.reference_data)
    read_time = parameters.read_time
    ma_table_reference = getattr(parameters, 'ma_table_reference', None)
    yield
    parameters.reference_data.clear()
    parameters.reference_data.update(reference_data)
    parameters.read_time = read_time
    parameters.ma_table_reference = ma_table_reference


@pytest.mark.crds
def test_crds_ma_table_sets_read_pattern_and_exposure_time(restore_romanisim_parameters):
    meta, read_pattern, exptime = _romanisim_metadata(COORD, SCA, BAND, MA_TABLE, DATE, usecrds=True)
    entry = parameters.ma_table_reference['roman']['science_tables'][f'SCI{MA_TABLE:04d}']
    assert read_pattern == [[int(r) for r in resultant] for resultant in entry['science_read_pattern']]
    assert meta['exposure']['read_pattern'] == read_pattern
    assert meta['exposure']['frame_time'] == pytest.approx(float(entry['frame_time']))
    assert exptime == pytest.approx(float(entry['frame_time']) * read_pattern[-1][-1])


@pytest.mark.crds
def test_crds_wcs_matches_the_one_romanisim_simulates_with(restore_romanisim_parameters):
    """romanisim builds its WCS from the full image metadata (defaults merged in and
    ``add_more_metadata`` run); ``_make_wcs`` builds it from the lighter metadata alone. The
    L3 path relies on the two agreeing."""
    from romanisim.models import wcs as romanisim_wcs

    ours = _make_wcs(COORD, SCA, BAND, MA_TABLE, None, DATE, PA_APER, usecrds=True)
    meta, _, _ = _romanisim_metadata(COORD, SCA, BAND, MA_TABLE, DATE, pa_aper=PA_APER, usecrds=True)
    romanisims = romanisim_wcs.get_wcs(_romanisim_image_model(meta, usecrds=True).meta, usecrds=True)
    for x, y in [(0, 0), (2043, 2043), (4087, 0), (0, 4087), (4087, 4087)]:
        a = ours.toWorld(galsim.PositionD(x, y))
        b = romanisims.toWorld(galsim.PositionD(x, y))
        assert a.distanceTo(b).deg * 3600 < 1e-6


@pytest.mark.crds
def test_crds_qe_inputs(restore_romanisim_parameters):
    from romanisim import image

    meta, _, _ = _romanisim_metadata(COORD, SCA, BAND, MA_TABLE, DATE, pa_aper=PA_APER, usecrds=True)
    flat, gain, area = _crds_qe_inputs(meta)
    assert np.shape(gain) == (DETECTOR_SIZE, DETECTOR_SIZE)
    assert area.shape == (DETECTOR_SIZE, DETECTOR_SIZE)
    assert np.all(np.isfinite(area)) and np.all(area > 0)
    qe = image.flat_to_qe(flat, gain, area=area)
    assert np.median(qe) == pytest.approx(1.0)


@pytest.mark.crds
def test_process_batch_l2_with_crds(tmp_path, restore_romanisim_parameters):
    """One lens through a CRDS-backed L2: its flux in the L2 must match the electrons put in,
    after the relative QE and the flat-field division in the ramp fit. This also catches the
    ramp fit using romanisim's built-in read time rather than the CRDS MA table's frame time,
    which scales every slope by their ratio (1.63 for MA table 17)."""
    from mejiro.galaxy_galaxy import SampleGG
    from mejiro.instruments.roman import Roman
    from mejiro.pipeline._05_romanisim import process_batch_l2
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils import util

    import romanisim.models.bandpass

    synth = SyntheticImage(strong_lens=SampleGG(), instrument=Roman(), band=BAND, fov_arcsec=5,
                           instrument_params={'detector': SCA, 'detector_position': (2043, 2043)})
    pickle_path = os.path.join(tmp_path, f'SyntheticImage_crds_00000001_{BAND}.pkl')
    util.pickle(pickle_path, synth)
    tile_size = synth.native_num_pix

    meta, _, exptime = _romanisim_metadata(COORD, SCA, BAND, MA_TABLE, DATE, pa_aper=PA_APER, usecrds=True)
    tile_r = tile_c = 20  # away from the detector edges
    task = ([(pickle_path, tile_r, tile_c)], SCA, BAND, 0, 1, str(tmp_path),
            exptime, 42, MA_TABLE, DATE, COORD, 0, 2, tile_size, True)
    process_batch_l2(task)
    assert os.path.exists(os.path.join(tmp_path, f'batch_complete_sca{SCA:02d}_{BAND}_batch0.txt')), \
        'the batch failed; see the logged traceback'

    im = util.unpickle(os.path.join(tmp_path, f'im_sca{SCA:02d}_{BAND}_batch0.pkl'))
    assert im.meta.ref_file.crds.context not in (None, '', 'N/A')
    data = np.asarray(im.data, dtype=np.float64)
    err = np.asarray(im.err, dtype=np.float64)

    rows = slice(tile_r * tile_size, (tile_r + 1) * tile_size)
    cols = slice(tile_c * tile_size, (tile_c + 1) * tile_size)
    background = np.median(data[rows.start - 100:rows.stop + 100, cols.start - 100:cols.stop + 100])
    measured = np.sum(data[rows, cols] - background)  # DN/s

    # what process_batch_l2 put in, and what romanisim's pipeline turns it into: electrons
    # times QE, divided by gain and flat in the ramp fit, i.e. electrons / (area * median(flat
    # * gain / area))
    abflux = romanisim.models.bandpass.get_abflux(BAND, SCA)
    smooth = np.asarray(util.smooth_pixels(synth.data), dtype=np.float64)
    electrons = smooth / smooth.sum() * synth.get_maggies() * abflux * exptime
    flat, gain, area = _crds_qe_inputs(meta)
    norm = np.median(np.asarray(flat) * gain / area)
    expected = np.sum(electrons / (area[rows, cols] * norm)) / exptime

    sigma = np.sqrt(np.sum(err[rows, cols] ** 2))
    assert abs(measured - expected) < 5 * sigma + 0.03 * expected, (measured, expected, sigma)
