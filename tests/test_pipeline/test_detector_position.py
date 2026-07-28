"""Tests for ``_detector_position``, which assigns each system its STPSF kernel.

This was an unseeded ``random.choice``, so the PSF a system was convolved with changed on
every step-04 run and ``--resume`` gave newly-rendered systems different kernels than the
pass that produced their neighbours. Datasets recorded the choice in ``.psfpos.json`` but
nothing could reproduce it. See docs/step04_oversampled_rendering.md.
"""
import numpy as np

from mejiro.pipeline._04_create_synthetic_images import _detector_position
from mejiro.utils import roman_util

# what divide_up_detector: 4 gives in the roman_data_challenge configs
POSITIONS = roman_util.divide_up_sca(4)


def test_is_deterministic():
    """The property that was false before: the same system gets the same PSF every run."""
    first = _detector_position('rung_1_00000006', 42, POSITIONS)
    assert first == _detector_position('rung_1_00000006', 42, POSITIONS)


def test_depends_on_both_seed_and_name():
    seeds = {_detector_position('lens_a', s, POSITIONS) for s in range(40)}
    names = {_detector_position(f'lens_{i}', 42, POSITIONS) for i in range(40)}
    assert len(seeds) > 1
    assert len(names) > 1


def test_returns_a_supplied_position():
    for i in range(200):
        assert _detector_position(f'lens_{i:08d}', 42, POSITIONS) in POSITIONS


def test_is_uniform_over_the_positions():
    """_05_romanisim's L2 path buckets systems by detector position and round-robins
    batches across those buckets, so a lopsided hash would leave some buckets starved."""
    counts = {}
    n = 16000
    for i in range(n):
        pos = _detector_position(f'lens_{i:08d}', 42, POSITIONS)
        counts[pos] = counts.get(pos, 0) + 1

    assert len(counts) == len(POSITIONS), 'some positions were never selected'
    expected = n / len(POSITIONS)
    # 16000 draws over 16 positions: expect 1000 each, sigma ~31, so 15% is ~5 sigma
    assert max(abs(c - expected) for c in counts.values()) < 0.15 * expected


def test_is_independent_of_ordering():
    """Keyed on the name, not list order, so the size-sort and limit subsampling step 04
    does cannot change which PSF a system gets."""
    names = [f'lens_{i:08d}' for i in range(50)]
    forward = {n: _detector_position(n, 42, POSITIONS) for n in names}
    backward = {n: _detector_position(n, 42, POSITIONS) for n in reversed(names)}
    assert forward == backward


def test_tracks_the_position_list_it_is_given():
    """divide_up_detector is a config knob; a system's position has to come from the list
    actually in force, not a cached one."""
    coarse = roman_util.divide_up_sca(2)
    for i in range(50):
        assert _detector_position(f'lens_{i}', 42, coarse) in coarse
