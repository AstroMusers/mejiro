"""Tests for ``subpixel_phase``, the L2 path's stand-in for a dither-derived sub-pixel phase.

Without it every system lands dead-center on a detector pixel: the step-04 grid is centered
on the deflector and the L2 tiling places tile centers at integer detector pixels. Measured
on 400 real rung-1 step-04 files, 97.5% had their brightest pixel at the exact center index.
That is the most favorable sampling phase there is and nothing like a real survey.
"""
import numpy as np

from mejiro.pipeline._05_romanisim import subpixel_phase


def test_is_deterministic():
    assert subpixel_phase(42, 'lens_00000006') == subpixel_phase(42, 'lens_00000006')


def test_depends_on_both_seed_and_name():
    assert subpixel_phase(42, 'a') != subpixel_phase(43, 'a')
    assert subpixel_phase(42, 'a') != subpixel_phase(42, 'b')


def test_axes_are_independent():
    # A single hash feeds both axes; slicing it wrong would make them equal or correlated.
    fx, fy = subpixel_phase(42, 'lens_00000006')
    assert fx != fy


def test_is_in_range():
    for i in range(1000):
        fx, fy = subpixel_phase(7, f'lens_{i:08d}')
        assert -0.5 <= fx < 0.5
        assert -0.5 <= fy < 0.5


def test_is_approximately_uniform():
    """Real surveys sample sub-pixel phase uniformly; a clumped hash would trade one
    systematic for another."""
    phases = np.array([subpixel_phase(42, f'lens_{i:08d}') for i in range(10000)])
    for axis in (0, 1):
        counts, _ = np.histogram(phases[:, axis], bins=10, range=(-0.5, 0.5))
        # 10000 draws in 10 bins: expect 1000 each, sigma ~30, so 10% is ~3 sigma
        assert np.abs(counts - 1000).max() < 100
        assert abs(phases[:, axis].mean()) < 0.02


def test_is_stable_across_ordering_and_bands():
    """The phase is keyed on seed + name only, so it cannot depend on which band is being
    processed or on where the system sits in a reordered work list -- both of which change
    between runs (step 04 size-sorts, step 05 subsamples per SCA)."""
    names = [f'lens_{i:08d}' for i in range(50)]
    forward = {n: subpixel_phase(42, n) for n in names}
    backward = {n: subpixel_phase(42, n) for n in reversed(names)}
    assert forward == backward
