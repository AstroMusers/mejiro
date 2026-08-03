"""Tests for ``bin_to_native``, the operator that turns an oversampled step-04 tile into a
detector-resolution tile at a chosen sub-pixel phase.

This is where the L3 dither registration fix lives (docs/l3_dither_registration.md). The
correctness claim is that the sub-pixel shift must happen on the *oversampled* grid: Roman
WFI at 0.11 arcsec/px is undersampled, so interpolating a native-resolution tile is
aliasing-limited exactly at the PSF core, while at 0.11/5 arcsec/px the scene is ~3x Nyquist
and the shift is exact. ``test_matches_exact_reference`` pins both halves of that claim
against a real STPSF kernel.
"""
import os

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, shift as ndshift

from mejiro.pipeline._05_romanisim import bin_to_native

OV = 5
N = 91

# SUB6 per-dither rounding residuals (dy, dx) in detector px, measured at the rung-1
# pointing and recorded in docs/l3_dither_registration.md. Dither 0 is exact by
# construction because the tile grid is laid out on integer dither-0 pixels.
SUB6_RESIDUALS = [
    (0.000, 0.000),
    (-0.164, -0.167),
    (0.171, 0.180),
    (0.408, -0.332),
    (0.247, 0.498),
    (0.085, 0.330),
]


def _block_sum(tile, oversample):
    n = tile.shape[0] // oversample
    return tile.reshape(n, oversample, n, oversample).sum(3).sum(1)


def _fourier_shift(tile, dy, dx):
    """Reference sub-pixel shift, independent of bin_to_native's own implementation."""
    return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(tile), (dy, dx))))


@pytest.fixture(scope='module')
def stpsf_kernel(test_data_dir):
    """Real Roman F129 PSF at oversample=5, i.e. 0.022 arcsec/px.

    A bare PSF is the hardest case for this operator: the core is the most undersampled
    structure the pipeline ever puts on the detector grid.
    """
    kernel = np.load(os.path.join(test_data_dir, 'F129_1_2048_2048_5_101.npy'))
    n = (kernel.shape[0] // OV) * OV  # trim to an exact multiple of the oversample
    kernel = kernel[:n, :n].astype(np.float64)
    return kernel / kernel.sum()


@pytest.fixture(scope='module')
def smooth_tile():
    """A band-limited oversampled tile: a Gaussian blob well resolved at 0.022 arcsec/px."""
    yy, xx = np.mgrid[:N * OV, :N * OV] - (N * OV - 1) / 2.0
    tile = np.exp(-(xx ** 2 + yy ** 2) / (2 * 12.0 ** 2))
    return tile / tile.sum()


def test_zero_phase_is_a_plain_block_sum(smooth_tile):
    # With no shift the operator must be exactly the detector pixel integral and nothing
    # else -- no interpolation, no resampling, bit-for-bit.
    np.testing.assert_array_equal(bin_to_native(smooth_tile, OV), _block_sum(smooth_tile, OV))


@pytest.mark.parametrize('frac_x,frac_y', [(0.3, -0.2), (0.5, 0.5), (-0.5, 0.017), (0.498, 0.247)])
def test_flux_is_conserved(smooth_tile, frac_x, frac_y):
    # A Fourier shift leaves the DC term untouched and binning is a block sum, so the total
    # is preserved to roundoff. Step 05 renormalizes to get_maggies() afterwards, but a leak
    # here would mean the *shape* had been altered, not just the scale.
    out = bin_to_native(smooth_tile, OV, frac_x, frac_y)
    assert out.sum() == pytest.approx(smooth_tile.sum(), rel=1e-12)


def test_shift_actually_changes_the_result(smooth_tile):
    unshifted = bin_to_native(smooth_tile, OV)
    shifted = bin_to_native(smooth_tile, OV, 0.4, 0.4)
    assert not np.allclose(unshifted, shifted)


@pytest.mark.parametrize('k', [1, 2, -3])
def test_integer_subpixel_shift_matches_an_exact_roll(smooth_tile, k):
    """A shift of k/oversample detector px is a whole number of oversampled samples, so it
    must reproduce an exact array roll. This is the check that the interpolation introduces
    nothing of its own."""
    got = bin_to_native(smooth_tile, OV, k / OV, 0.0)
    expected = _block_sum(np.roll(smooth_tile, k, axis=1), OV)
    assert np.abs(got - expected).max() / expected.max() < 1e-12

    got = bin_to_native(smooth_tile, OV, 0.0, k / OV)
    expected = _block_sum(np.roll(smooth_tile, k, axis=0), OV)
    assert np.abs(got - expected).max() / expected.max() < 1e-12


def test_shift_direction_matches_place_tile(smooth_tile):
    """frac_x must move flux along axis 1 and frac_y along axis 0, in the same direction
    ``_place_tile`` moves an integer offset.

    A sign or axis swap here is silent -- it just re-misregisters the dithers -- and is
    exactly the class of defect behind docs/l3_cutout_orientation.md.
    """
    base = bin_to_native(smooth_tile, OV)
    n = base.shape[0]
    yy, xx = np.mgrid[:n, :n]

    def centroid(a):
        return (a * xx).sum() / a.sum(), (a * yy).sum() / a.sum()

    cx0, cy0 = centroid(base)
    cx, cy = centroid(bin_to_native(smooth_tile, OV, 0.4, 0.0))
    assert cx - cx0 == pytest.approx(0.4, abs=0.02)
    assert cy - cy0 == pytest.approx(0.0, abs=0.02)

    cx, cy = centroid(bin_to_native(smooth_tile, OV, 0.0, 0.4))
    assert cx - cx0 == pytest.approx(0.0, abs=0.02)
    assert cy - cy0 == pytest.approx(0.4, abs=0.02)


def test_matches_exact_reference_where_a_native_shift_does_not(stpsf_kernel):
    """On a real Roman PSF, binning at the true phase must reproduce the exact answer, and
    the native-resolution alternative must not.

    The reference is the kernel displaced on the oversampled grid and then binned. The
    second half of the assertion is what justifies paying for oversampled step-04 output at
    all: shifting the already-binned tile is the cheap fix, and it is wrong at the percent
    level on the core.
    """
    ov = OV
    n = stpsf_kernel.shape[0] // ov
    native = _block_sum(stpsf_kernel, ov)

    worst_native_err = 0.0
    for dy, dx in SUB6_RESIDUALS:
        reference = _block_sum(_fourier_shift(stpsf_kernel, dy * ov, dx * ov), ov)

        got = bin_to_native(stpsf_kernel, ov, dx, dy)
        assert np.abs(got - reference).max() / reference.max() < 1e-9

        native_shift = ndshift(native, (dy, dx), order=3, mode='nearest')
        worst_native_err = max(
            worst_native_err, np.abs(native_shift - reference).max() / reference.max()
        )

    # measured ~0.11 for these residuals; assert only that it is far above the tolerance
    # bin_to_native meets, so the test documents the gap rather than a specific number
    assert worst_native_err > 0.01


def test_oversample_one_is_an_identity():
    tile = np.arange(9.0).reshape(3, 3)
    np.testing.assert_array_equal(bin_to_native(tile, 1), tile)


def test_oversample_one_rejects_a_sub_pixel_phase():
    # There is no valid way to apply a sub-pixel phase to a detector-resolution tile;
    # doing it anyway is the aliasing-limited operation this function exists to avoid.
    with pytest.raises(ValueError, match='oversample=1'):
        bin_to_native(np.zeros((3, 3)), 1, 0.25, 0.0)


def test_rejects_a_tile_that_is_not_a_multiple_of_oversample():
    with pytest.raises(ValueError, match='multiple of oversample'):
        bin_to_native(np.zeros((92, 92)), 5)


def test_rejects_a_non_square_tile():
    with pytest.raises(ValueError, match='square'):
        bin_to_native(np.zeros((455, 450)), 5)
