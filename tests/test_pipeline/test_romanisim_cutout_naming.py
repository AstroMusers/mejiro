"""Tests for romanisim exposure cutout filename handling.

Regression coverage for the bug where lightweight ``.npz`` SyntheticImage inputs
produced ``Exposure_..._{band}.npz.npy`` cutouts that the HDF5 export step could not
locate (it looked for ``Exposure_..._{band}.npy``).
"""
from mejiro.pipeline._05_romanisim import exposure_cutout_name


def test_exposure_cutout_name_from_npz():
    # .npz input must not leak its extension into the .npy cutout name
    name = exposure_cutout_name('05/sca01/SyntheticImage_foo_00000006_F158.npz')
    assert name == 'Exposure_foo_00000006_F158.npy'


def test_exposure_cutout_name_from_pkl():
    # legacy .pkl input keeps producing the same canonical name
    name = exposure_cutout_name('SyntheticImage_foo_00000006_F158.pkl')
    assert name == 'Exposure_foo_00000006_F158.npy'
