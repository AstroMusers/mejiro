"""Tests for lightweight step-03 serialization: stripping the pyHalo realization.

Mirrors the lightweight tests in ``test_synthetic_image.py``. Uses sample lenses
plus a small, fast CDM realization (``cone_opening_angle_arcsec=5``, as in
``test_pyhalo.py``) so nothing touches real pipeline data.
"""
import os
import pickle

import numpy as np
import pytest
from pyHalo.preset_models import preset_model_from_name

from mejiro.analysis import lensing
from mejiro.galaxy_galaxy import Sample1, SampleGG, SampleSL2S, SampleBELLS
from mejiro.instruments.roman import Roman
from mejiro.synthetic_image import LightweightStrongLens, SyntheticImage
from mejiro.utils import util


def _cdm_realization(strong_lens):
    CDM = preset_model_from_name('CDM')
    return CDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2),
               cone_opening_angle_arcsec=5,
               log_m_host=np.log10(strong_lens.get_main_halo_mass()))


@pytest.mark.parametrize('model,expected', [('CDM', 'CDM'), ('WDM', 'WDM')])
def test_substructure_flag_derivation(model, expected):
    """substructure_flag() extracts the model abbreviation from a realization."""
    sl = SampleGG()
    Model = preset_model_from_name(model)
    kwargs = dict(cone_opening_angle_arcsec=5, log_m_host=np.log10(sl.get_main_halo_mass()))
    if model == 'WDM':
        kwargs['log_mc'] = 7
    realization = Model(round(sl.z_lens, 2), round(sl.z_source, 2), **kwargs)
    assert lensing.substructure_flag(realization) == expected


def test_strip_realization_preserves_lensing_inputs():
    """Stripping drops the realization to the sentinel but leaves the baked-in
    lens model (what step 04 ray-shoots from) and the macromodel attrs intact."""
    sl = SampleGG()
    sl.add_realization(_cdm_realization(sl))

    kwargs_lens_before = [dict(d) for d in sl.kwargs_lens]
    lens_model_list_before = list(sl.lens_model_list)
    lens_redshift_list_before = list(sl.lens_redshift_list)

    lensing.strip_realization(sl)

    # sentinel preserves `realization is not None` (has_realization semantics)
    assert sl.realization == lensing.LIGHTWEIGHT_REALIZATION
    assert sl.realization is not None
    assert sl.substructure_flag == 'CDM'

    # image-generation inputs untouched
    assert sl.kwargs_lens == kwargs_lens_before
    assert sl.lens_model_list == lens_model_list_before
    assert sl.lens_redshift_list == lens_redshift_list_before

    # macromodel attrs (used by the GalaxyGalaxy.get_image_positions fast branch) survive
    assert sl.lens_model_macromodel is not None
    assert sl.lens_model_list_macromodel is not None
    assert sl.kwargs_lens_macromodel is not None


def test_strip_realization_shrinks_pickle():
    """The stripped pickle is much smaller, and the sentinel + flag round-trip."""
    sl = SampleGG()
    realization = _cdm_realization(sl)
    sl.add_realization(realization)

    size_full = len(pickle.dumps(sl))
    realization_size = len(pickle.dumps(realization))
    lensing.strip_realization(sl)
    size_light = len(pickle.dumps(sl))

    # the realization object dominates the pickle
    assert size_light < 0.5 * size_full
    assert realization_size > size_light

    restored = pickle.loads(pickle.dumps(sl))
    assert restored.realization == lensing.LIGHTWEIGHT_REALIZATION
    assert restored.substructure_flag == 'CDM'


@pytest.mark.parametrize('strong_lens', [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_get_image_positions_unchanged_after_strip(strong_lens):
    """get_image_positions(ignore_substructure=True) is what step 04's adaptive
    grid calls. The truthy sentinel must keep it on the macromodel branch
    (galaxy_galaxy.py:65), giving identical positions to the un-stripped lens."""
    strong_lens.add_realization(_cdm_realization(strong_lens))

    x_full, y_full = strong_lens.get_image_positions(ignore_substructure=True)
    lensing.strip_realization(strong_lens)
    x_light, y_light = strong_lens.get_image_positions(ignore_substructure=True)

    np.testing.assert_allclose(np.sort(x_light), np.sort(x_full))
    np.testing.assert_allclose(np.sort(y_light), np.sort(y_full))


def test_stored_flag_matches_derived():
    """The flag stored at strip time equals what make_substructure_csv would
    derive from a full realization, so the answer key is identical either way."""
    sl = SampleGG()
    realization = _cdm_realization(sl)
    derived = lensing.substructure_flag(realization)
    sl.add_realization(realization)
    lensing.strip_realization(sl)
    assert sl.substructure_flag == derived


def test_sentinel_makes_save_lightweight_report_has_realization(tmp_path):
    """A lens carrying the strip sentinel must serialize as has_realization=True
    through the step-04 lightweight synthetic image."""
    si = SyntheticImage(
        strong_lens=Sample1(),
        instrument=Roman(),
        band='F129',
        fov_arcsec=5,
        instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
    )
    si.strong_lens.realization = lensing.LIGHTWEIGHT_REALIZATION
    path = str(tmp_path / 'si.npz')
    si.save_lightweight(path)

    loaded = util.load_synthetic_image(path)
    assert loaded.strong_lens.realization is not None


def test_sentinel_agrees_with_lightweight_strong_lens():
    """The strip sentinel value must match the literal LightweightStrongLens
    attaches on load, so the two stay in sync."""
    meta = {'band': 'F129', 'lens': {
        'name': 'x', 'z_lens': 0.5, 'z_source': 2.0, 'has_realization': True,
        'main_halo_mass': 1e13, 'einstein_radius': 1.0, 'velocity_dispersion': 250.0,
        'magnification': 5.0, 'lens_magnitude': 20.0, 'source_magnitude': 22.0,
        'lensed_source_magnitude': 21.0,
    }}
    assert LightweightStrongLens(meta).realization == lensing.LIGHTWEIGHT_REALIZATION


def test_compaction_roundtrip(tmp_path):
    """compact_03_realizations strips an existing full pickle in place, atomically,
    and is idempotent on re-run."""
    from mejiro.pipeline import compact_03_realizations as compaction

    sl = SampleGG()
    sl.add_realization(_cdm_realization(sl))
    path = str(tmp_path / 'lens_test.pkl')
    util.pickle(path, sl)
    size_before = os.path.getsize(path)

    status, saved = compaction._compact_one(path)
    assert status == 'compacted'
    assert saved > 0

    reloaded = util.unpickle(path)
    assert reloaded.realization == lensing.LIGHTWEIGHT_REALIZATION
    assert reloaded.substructure_flag == 'CDM'
    assert len(reloaded.kwargs_lens) == len(reloaded.lens_model_list) == len(reloaded.lens_redshift_list)
    assert os.path.getsize(path) < size_before

    # idempotent: a second pass detects the sentinel and skips
    status2, saved2 = compaction._compact_one(path)
    assert status2 == 'skip'
    assert saved2 == 0
