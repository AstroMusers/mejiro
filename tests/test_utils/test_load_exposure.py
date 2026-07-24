"""Loader-level tests for ``mejiro.utils.util.load_exposure``.

These exercise the .pkl/.npz/.npy dispatch and schema-version handling without
constructing a real Exposure (which pulls in lenstronomy + galsim + sample lenses).
Mirrors ``tests/test_utils/test_load_synthetic_image.py``.
"""
import json

import numpy as np
import pytest

from mejiro.exposure import EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION, LightweightExposure
from mejiro.utils import util


def _write_lightweight_npz(path, data, meta, lens_data=None, source_data=None):
    meta_bytes = json.dumps(meta).encode('utf-8')
    arrays = {
        'data': data.astype(np.float32),
        'meta': np.frombuffer(meta_bytes, dtype=np.uint8),
    }
    if lens_data is not None:
        arrays['lens_data'] = lens_data.astype(np.float32)
    if source_data is not None:
        arrays['source_data'] = source_data.astype(np.float32)
    with open(path, 'wb') as fh:
        np.savez(fh, **arrays)


def _valid_meta(band='F129'):
    return {
        'schema_version': EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION,
        'band': band,
        'instrument_name': 'Roman',
        'num_pix': 45,
        'pixel_scale': 0.11,
        'exposure_time': 146.0,
        'engine': 'galsim',
        'pieces': False,
        'lens': {'name': 'test_lens', 'z_lens': 0.5, 'z_source': 2.0},
    }


def test_load_exposure_dispatch_pkl(tmp_path):
    """``.pkl`` path falls through to ``unpickle`` and returns the stored object."""
    sentinel = {'kind': 'sentinel', 'value': 7}
    path = tmp_path / 'thing.pkl'
    util.pickle(str(path), sentinel)

    loaded = util.load_exposure(str(path))
    assert loaded == sentinel


def test_load_exposure_dispatch_npz(tmp_path):
    """``.npz`` path returns a ``LightweightExposure`` with parsed metadata."""
    data = np.linspace(0.0, 1.0, 25).reshape(5, 5)
    meta = _valid_meta()
    path = tmp_path / 'lightweight.npz'
    _write_lightweight_npz(str(path), data, meta)

    loaded = util.load_exposure(str(path))
    assert isinstance(loaded, LightweightExposure)
    np.testing.assert_array_equal(loaded.data, data.astype(np.float32))
    assert loaded.band == 'F129'
    assert loaded.instrument_name == 'Roman'
    assert loaded.num_pix == meta['num_pix']
    assert loaded.pixel_scale == pytest.approx(meta['pixel_scale'])
    assert loaded.exposure_time == pytest.approx(meta['exposure_time'])
    assert loaded.engine == 'galsim'
    assert loaded.name == meta['lens']['name']
    assert loaded.z_lens == pytest.approx(meta['lens']['z_lens'])
    assert loaded.z_source == pytest.approx(meta['lens']['z_source'])
    assert loaded.lens_data is None
    assert loaded.source_data is None
    assert loaded.synthetic_image is None


def test_load_exposure_dispatch_npz_with_pieces(tmp_path):
    """``lens_data``/``source_data`` arrays round-trip when present in the .npz."""
    data = np.ones((4, 4))
    lens = np.full((4, 4), 2.0)
    source = np.full((4, 4), 3.0)
    meta = _valid_meta()
    meta['pieces'] = True
    path = tmp_path / 'pieces.npz'
    _write_lightweight_npz(str(path), data, meta, lens_data=lens, source_data=source)

    loaded = util.load_exposure(str(path))
    np.testing.assert_array_equal(loaded.lens_data, lens.astype(np.float32))
    np.testing.assert_array_equal(loaded.source_data, source.astype(np.float32))


def test_load_exposure_dispatch_npy(tmp_path):
    """``.npy`` path returns a data-only ``LightweightExposure`` (romanisim cutout)."""
    data = np.arange(9, dtype=float).reshape(3, 3)
    path = tmp_path / 'romanisim.npy'
    np.save(str(path), data)

    loaded = util.load_exposure(str(path))
    assert isinstance(loaded, LightweightExposure)
    np.testing.assert_array_equal(loaded.data, data)
    # bare arrays carry no metadata or pieces
    assert loaded.lens_data is None
    assert loaded.source_data is None
    assert loaded.band is None
    assert loaded.exposure_time is None


def test_load_exposure_rejects_unknown_schema(tmp_path):
    """An unknown schema_version surfaces a clear ValueError rather than silently
    handing back stale fields."""
    meta = _valid_meta()
    meta['schema_version'] = EXPOSURE_LIGHTWEIGHT_SCHEMA_VERSION + 99
    path = tmp_path / 'bad_schema.npz'
    _write_lightweight_npz(str(path), np.zeros((3, 3)), meta)

    with pytest.raises(ValueError, match='schema_version'):
        util.load_exposure(str(path))


def test_load_exposure_rejects_missing_schema(tmp_path):
    """A meta blob without ``schema_version`` also fails fast."""
    meta = _valid_meta()
    del meta['schema_version']
    path = tmp_path / 'no_schema.npz'
    _write_lightweight_npz(str(path), np.zeros((3, 3)), meta)

    with pytest.raises(ValueError, match='schema_version'):
        util.load_exposure(str(path))
