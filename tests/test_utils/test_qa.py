"""Tests for ``mejiro.utils.qa``, the exposure check run on romanisim HDF5 exports.

The pixel patterns below are the real signatures from the co-add defect in
docs/l3_negative_drizzle_weights.md: an isolated spike at the deflector core, a near-zero
pixel beside it, and negative surface brightness.
"""
import h5py
import numpy as np
import pytest

from mejiro.utils import qa


def _smooth_exposure(num_pix=45):
    """A clean, smoothly peaked exposure on a ~1 MJy/sr background, as an L3 co-add looks."""
    y, x = np.mgrid[0:num_pix, 0:num_pix]
    c = num_pix // 2
    return (1.0 + 40.0 * np.exp(-((y - c) ** 2 + (x - c) ** 2) / 8.0)).astype(np.float32)


def _write_h5(path, exposures):
    """exposures: {(uid, band): array}"""
    with h5py.File(path, 'w') as f:
        images = f.create_group('images')
        for (uid, band), data in exposures.items():
            group = images.require_group(f'strong_lens_{uid}')
            group.create_dataset(f'exposure_{uid}_{band}', data=data)


def test_neighbor_ratio_is_one_on_a_flat_field():
    ratio = qa.neighbor_ratio(np.full((9, 9), 3.0))
    assert ratio.shape == (7, 7)
    assert np.allclose(ratio, 1.0)


def test_clean_exposures_pass(tmp_path):
    path = tmp_path / 'clean.h5'
    _write_h5(path, {('00000000', b): _smooth_exposure() for b in ('F106', 'F129', 'F158')})

    assert qa.find_corrupted_exposures(path) == []
    qa.check_exposures(path)  # must not raise


@pytest.mark.parametrize('value, expected', [
    (236.3, 'neighbour median'),   # spike, as in uid 00000005 F129
    (0.01, 'neighbour median'),    # near-zero beside a bright core
])
def test_ratio_outliers_are_flagged(tmp_path, value, expected):
    data = _smooth_exposure()
    data[20, 21] = value
    path = tmp_path / 'ratio.h5'
    _write_h5(path, {('00000001', 'F129'): data})

    findings = qa.find_corrupted_exposures(path)
    assert len(findings) == 1
    uid, band, reason = findings[0]
    assert (uid, band) == ('00000001', 'F129')
    assert expected in reason and '(20, 21)' in reason


def test_negative_pixel_is_flagged(tmp_path):
    data = _smooth_exposure()
    data[22, 23] = -176.3
    path = tmp_path / 'negative.h5'
    _write_h5(path, {('00000002', 'F158'): data})

    findings = qa.find_corrupted_exposures(path)
    assert len(findings) == 1
    assert 'negative pixel' in findings[0][2]

    with pytest.raises(ValueError, match='1 corrupted exposure'):
        qa.check_exposures(path)


def test_non_finite_pixel_is_flagged(tmp_path):
    data = _smooth_exposure()
    data[10, 10] = np.nan
    path = tmp_path / 'nan.h5'
    _write_h5(path, {('00000003', 'F106'): data})

    assert 'non-finite' in qa.find_corrupted_exposures(path)[0][2]


def test_check_reports_every_offending_exposure(tmp_path):
    spike, negative = _smooth_exposure(), _smooth_exposure()
    spike[20, 21] = 500.0
    negative[22, 23] = -5.0
    path = tmp_path / 'mixed.h5'
    _write_h5(path, {
        ('00000004', 'F106'): _smooth_exposure(),
        ('00000004', 'F129'): spike,
        ('00000005', 'F158'): negative,
    })

    findings = qa.find_corrupted_exposures(path)
    assert {(uid, band) for uid, band, _ in findings} == {('00000004', 'F129'), ('00000005', 'F158')}
    with pytest.raises(ValueError, match='2 corrupted exposure\\(s\\) across 2 system\\(s\\)'):
        qa.check_exposures(path)


def test_synthetic_image_datasets_are_ignored(tmp_path):
    """Only exposure_* datasets are checked; synthetic images are noiseless and unaffected."""
    path = tmp_path / 'synth.h5'
    with h5py.File(path, 'w') as f:
        group = f.create_group('images').create_group('strong_lens_00000006')
        group.create_dataset('exposure_00000006_F129', data=_smooth_exposure())
        bad = _smooth_exposure()
        bad[20, 21] = -99.0
        group.create_dataset('synthetic_image_00000006_F129', data=bad)

    assert qa.find_corrupted_exposures(path) == []
