"""Loader-level tests for ``mejiro.utils.util.load_synthetic_image``.

These exercise the .pkl/.npz dispatch and schema-version handling without
constructing a real SyntheticImage (which pulls in lenstronomy + sample lenses).
"""
import json

import numpy as np
import pytest

from mejiro.synthetic_image import (
    LIGHTWEIGHT_SCHEMA_VERSION,
    LightweightSyntheticImage,
)
from mejiro.utils import util


def _write_lightweight_npz(path, data, meta):
    meta_bytes = json.dumps(meta).encode('utf-8')
    with open(path, 'wb') as fh:
        np.savez(
            fh,
            data=data.astype(np.float32),
            meta=np.frombuffer(meta_bytes, dtype=np.uint8),
        )


def _valid_meta(band='F129'):
    return {
        'schema_version': LIGHTWEIGHT_SCHEMA_VERSION,
        'band': band,
        'pixel_scale': 0.11,
        'fov_arcsec': 8.03,
        'num_pix': 73,
        'instrument_name': 'Roman',
        'instrument_params': {'detector': 7, 'detector_position': [2044, 2044]},
        'magnitude_zeropoint': 26.5,
        'lens': {
            'name': 'test_lens',
            'z_lens': 0.34,
            'z_source': 1.82,
            'has_realization': False,
            'main_halo_mass': 2.5e13,
            'einstein_radius': 1.12,
            'velocity_dispersion': 215.0,
            'magnification': 3.7,
            'lens_magnitude': 21.4,
            'source_magnitude': 25.1,
            'lensed_source_magnitude': 23.7,
        },
    }


def test_load_synthetic_image_dispatch_pkl(tmp_path):
    """``.pkl`` path falls through to ``unpickle`` and returns the stored object."""
    sentinel = {'kind': 'sentinel', 'value': 42}
    path = tmp_path / 'thing.pkl'
    util.pickle(str(path), sentinel)

    loaded = util.load_synthetic_image(str(path))
    assert loaded == sentinel


def test_load_synthetic_image_dispatch_npz(tmp_path):
    """``.npz`` path returns a ``LightweightSyntheticImage`` with parsed metadata."""
    data = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    meta = _valid_meta()
    path = tmp_path / 'lightweight.npz'
    _write_lightweight_npz(str(path), data, meta)

    loaded = util.load_synthetic_image(str(path))
    assert isinstance(loaded, LightweightSyntheticImage)
    np.testing.assert_array_equal(loaded.data, data.astype(np.float32))
    assert loaded.band == meta['band']
    assert loaded.pixel_scale == pytest.approx(meta['pixel_scale'])
    assert loaded.fov_arcsec == pytest.approx(meta['fov_arcsec'])
    assert loaded.instrument_params['detector'] == 7
    assert loaded.instrument_params['detector_position'] == (2044, 2044)
    assert loaded.strong_lens.realization is None
    assert loaded.strong_lens.z_lens == pytest.approx(meta['lens']['z_lens'])
    assert loaded.strong_lens.get_einstein_radius() == pytest.approx(meta['lens']['einstein_radius'])
    assert loaded.strong_lens.get_lens_magnitude(meta['band']) == pytest.approx(
        meta['lens']['lens_magnitude']
    )


def test_load_synthetic_image_has_realization_sentinel(tmp_path):
    """``has_realization=True`` round-trips so the substructure flag survives."""
    meta = _valid_meta()
    meta['lens']['has_realization'] = True
    path = tmp_path / 'with_sub.npz'
    _write_lightweight_npz(str(path), np.zeros((3, 3)), meta)

    loaded = util.load_synthetic_image(str(path))
    assert loaded.strong_lens.realization is not None


def test_load_synthetic_image_rejects_unknown_schema(tmp_path):
    """An unknown schema_version surfaces a clear ValueError rather than silently
    handing back stale fields."""
    meta = _valid_meta()
    meta['schema_version'] = LIGHTWEIGHT_SCHEMA_VERSION + 99
    path = tmp_path / 'bad_schema.npz'
    _write_lightweight_npz(str(path), np.zeros((3, 3)), meta)

    with pytest.raises(ValueError, match='schema_version'):
        util.load_synthetic_image(str(path))


def test_load_synthetic_image_rejects_missing_schema(tmp_path):
    """A meta blob without ``schema_version`` also fails fast."""
    meta = _valid_meta()
    del meta['schema_version']
    path = tmp_path / 'no_schema.npz'
    _write_lightweight_npz(str(path), np.zeros((3, 3)), meta)

    with pytest.raises(ValueError, match='schema_version'):
        util.load_synthetic_image(str(path))
