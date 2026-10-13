import importlib
from types import SimpleNamespace

import pytest


def test_make_single_nfw_subhalo_preserves_pyhalo_arguments(monkeypatch):
    pyhalo = pytest.importorskip('pyHalo')
    single_realization = importlib.import_module('pyHalo.single_realization')
    captured = {}

    class FakeSingleHalo:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(single_realization, 'SingleHalo', FakeSingleHalo)
    from mejiro.utils.pyhalo_util import make_single_nfw_subhalo

    lens = SimpleNamespace(z_lens=0.5, z_source=1.5, cosmo=object())
    lens_cosmo = object()
    halo_model = {'concentration_model': object()}

    result = make_single_nfw_subhalo(lens, 1e8, 0.1, -0.2, lens_cosmo, halo_model)

    assert isinstance(result, FakeSingleHalo)
    assert captured == {
        'halo_mass': 1e8,
        'x': 0.1,
        'y': -0.2,
        'mdef': 'NFW',
        'z': 0.5,
        'zlens': 0.5,
        'zsource': 1.5,
        'subhalo_flag': True,
        'kwargs_halo_model': halo_model,
        'astropy_instance': lens.cosmo,
        'lens_cosmo': lens_cosmo,
    }