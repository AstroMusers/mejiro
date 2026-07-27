"""Tests for the lenstronomy kwargs builders in ``mejiro.utils.lenstronomy_util``.

``degrade`` exists because lenstronomy folds the detector pixel response into any kernel
supplied with ``point_source_supersampling_factor > 1``: ``kernel_util.degrade_kernel`` is
an exact NxN box average. That is what a native-resolution pixel grid wants, and exactly
what an oversampled one must not have -- see ``SyntheticImage(oversample=...)``.
"""
import numpy as np
import pytest
from lenstronomy.Data.psf import PSF

from mejiro.utils.lenstronomy_util import get_pixel_psf_kwargs


def _kernel(n=25):
    yy, xx = np.mgrid[:n, :n] - n // 2
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * 3.0 ** 2))
    return k / k.sum()


def test_degrade_true_is_the_default_and_keeps_the_supersampling_factor():
    kwargs = get_pixel_psf_kwargs(_kernel(), 5)
    assert kwargs['psf_type'] == 'PIXEL'
    assert kwargs['point_source_supersampling_factor'] == 5


def test_degrade_false_tells_lenstronomy_to_use_the_kernel_as_is():
    kwargs = get_pixel_psf_kwargs(_kernel(), 5, degrade=False)
    assert kwargs['point_source_supersampling_factor'] == 1


def test_degrade_false_leaves_the_kernel_untouched_in_lenstronomy():
    """The behavioural difference the flag exists for: with degrade=True lenstronomy hands
    back a box-averaged kernel one fifth the size; with degrade=False it hands back what it
    was given."""
    kernel = _kernel()

    degraded = PSF(**get_pixel_psf_kwargs(kernel, 5)).kernel_point_source
    assert degraded.shape[0] < kernel.shape[0]

    as_is = PSF(**get_pixel_psf_kwargs(kernel, 5, degrade=False)).kernel_point_source
    np.testing.assert_allclose(as_is, kernel)


def test_degrading_adds_exactly_one_pixel_response():
    """Pins why the flag matters numerically: the degraded kernel's second moment is the
    original's plus the variance of an ``oversample``-wide box, i.e. one detector pixel
    integral. Convolving an already pixel-integrated scene with it applies that twice."""
    ov = 5
    n = 25 * ov
    yy, xx = np.mgrid[:n, :n] - n // 2
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * 15.0 ** 2))
    kernel /= kernel.sum()

    def second_moment(k):
        m = k.shape[0]
        _, x = np.mgrid[:m, :m] - m // 2
        w = k / k.sum()
        return (w * x ** 2).sum()

    degraded = PSF(**get_pixel_psf_kwargs(kernel, ov)).kernel_point_source
    box_variance = (ov ** 2 - 1) / 12 / ov ** 2
    assert second_moment(degraded) == pytest.approx(
        second_moment(kernel) / ov ** 2 + box_variance, rel=1e-4
    )
