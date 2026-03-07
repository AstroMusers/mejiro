import os
import numpy as np
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.galaxy_galaxy import SampleGG
from mejiro.instruments.roman import Roman
from mejiro.engines.stpsf_engine import STPSFEngine
from lenstronomy.Util import data_util


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(mejiro.__file__)), 'tests', 'test_data')


def _make_noiseless_exposure(strong_lens, band='F129', exposure_time=146, pieces=True):
    """Helper to create a noiseless exposure for flux testing."""
    detector = 'SCA01'
    detector_position = (2048, 2048)

    kwargs_psf = STPSFEngine.get_roman_psf_kwargs(
        band, detector, detector_position, oversample=5, num_pix=101,
        check_cache=True, psf_cache_dir=TEST_DATA_DIR, verbose=False)

    synthetic_image = SyntheticImage(
        strong_lens=strong_lens,
        instrument=Roman(),
        band=band,
        fov_arcsec=5,
        instrument_params={'detector': detector, 'detector_position': detector_position},
        kwargs_numerics={},
        kwargs_psf=kwargs_psf,
        pieces=pieces,
        verbose=False)

    # disable all noise and detector effects
    engine_params = {
        'sky_background': False,
        'detector_effects': False,
    }
    exposure = Exposure(
        synthetic_image,
        exposure_time=exposure_time,
        engine='galsim',
        engine_params=engine_params,
        verbose=False)

    return synthetic_image, exposure


def test_noiseless_exposure_flux():
    """
    Test that a noiseless exposure conserves flux from the synthetic image.

    With all noise and detector effects disabled, the exposure should be
    approximately synthetic_image.image * exposure_time, with only integer
    quantization error.
    """
    strong_lens = SampleGG()
    exposure_time = 146
    synthetic_image, exposure = _make_noiseless_exposure(strong_lens, exposure_time=exposure_time)

    expected_total = np.sum(synthetic_image.image) * exposure_time
    actual_total = np.sum(exposure.exposure)

    # quantization error: at most 0.5 per pixel
    max_quantization_error = 0.5 * synthetic_image.num_pix ** 2
    assert abs(actual_total - expected_total) < max_quantization_error, \
        f'Total flux mismatch: expected {expected_total:.1f}, got {actual_total:.1f}'


def test_pieces_sum_to_total():
    """
    Test that lens + source surface brightness components sum to the total image.
    """
    strong_lens = SampleGG()
    synthetic_image, exposure = _make_noiseless_exposure(strong_lens)

    # at the SyntheticImage level, lens + source should equal total
    pieces_sum = synthetic_image.lens_surface_brightness + synthetic_image.source_surface_brightness
    np.testing.assert_allclose(synthetic_image.image, pieces_sum, rtol=1e-10,
                               err_msg='Lens + source surface brightness does not sum to total image')


def test_lens_magnitude():
    """
    Test that the lens flux in the synthetic image recovers the input magnitude.

    Convert the total lens flux (sum of lens surface brightness) back to a magnitude
    using the zero-point, and compare to the input lens magnitude.
    """
    strong_lens = SampleGG()
    band = 'F129'
    expected_lens_mag = strong_lens.get_lens_magnitude(band)

    synthetic_image, _ = _make_noiseless_exposure(strong_lens, band=band)

    # sum the lens surface brightness to get total lens flux in counts/sec
    lens_flux_cps = np.sum(synthetic_image.lens_surface_brightness)
    assert lens_flux_cps > 0, 'Lens flux should be positive'

    # convert back to magnitude
    measured_mag = data_util.cps2magnitude(lens_flux_cps, synthetic_image.magnitude_zeropoint)

    # tolerance accounts for finite FOV truncation of the Sersic profile
    assert abs(measured_mag - expected_lens_mag) < 0.2, \
        f'Lens magnitude mismatch: expected {expected_lens_mag}, measured {measured_mag:.3f}'


def test_source_magnitude():
    """
    Test that the lensed source flux recovers a magnitude brighter than the
    unlensed source (due to magnification).
    """
    strong_lens = SampleGG()
    band = 'F129'
    unlensed_source_mag = strong_lens.get_source_magnitude(band)

    synthetic_image, _ = _make_noiseless_exposure(strong_lens, band=band)

    # sum the source surface brightness (this is the lensed source)
    source_flux_cps = np.sum(synthetic_image.source_surface_brightness)
    assert source_flux_cps > 0, 'Source flux should be positive'

    measured_mag = data_util.cps2magnitude(source_flux_cps, synthetic_image.magnitude_zeropoint)

    # lensed source should be brighter (lower magnitude) than unlensed
    assert measured_mag < unlensed_source_mag, \
        f'Lensed source (mag={measured_mag:.3f}) should be brighter than unlensed (mag={unlensed_source_mag})'


def test_total_image_maggies():
    """
    Test that SyntheticImage.get_maggies() returns a consistent value.

    The total maggies should be the sum of lens maggies and lensed source maggies.
    """
    strong_lens = SampleGG()
    band = 'F129'

    synthetic_image, _ = _make_noiseless_exposure(strong_lens, band=band)

    total_maggies = synthetic_image.get_maggies()
    assert total_maggies > 0, 'Total maggies should be positive'

    # compute maggies from individual components
    lens_flux_cps = np.sum(synthetic_image.lens_surface_brightness)
    source_flux_cps = np.sum(synthetic_image.source_surface_brightness)

    lens_mag = data_util.cps2magnitude(lens_flux_cps, synthetic_image.magnitude_zeropoint)
    source_mag = data_util.cps2magnitude(source_flux_cps, synthetic_image.magnitude_zeropoint)

    lens_maggies = 10 ** (-0.4 * lens_mag)
    source_maggies = 10 ** (-0.4 * source_mag)

    # total maggies = sum of component maggies (flux is additive in linear space)
    expected_maggies = (lens_maggies + source_maggies).item()

    np.testing.assert_allclose(total_maggies, expected_maggies, rtol=1e-5,
                               err_msg='Total maggies does not equal sum of lens + source maggies')


def test_exposure_counts_match_synthetic_flux():
    """
    Test that the noiseless exposure total counts equal the synthetic image
    get_flux() * exposure_time, accounting for quantization.
    """
    strong_lens = SampleGG()
    exposure_time = 146
    synthetic_image, exposure = _make_noiseless_exposure(
        strong_lens, exposure_time=exposure_time, pieces=False)

    expected_counts = synthetic_image.get_flux() * exposure_time
    actual_counts = np.sum(exposure.exposure)

    relative_error = abs(actual_counts - expected_counts) / expected_counts
    assert relative_error < 1e-3, \
        f'Relative flux error {relative_error:.6f} exceeds tolerance of 1e-3'
