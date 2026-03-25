import os
import numpy as np
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.galaxy_galaxy import SampleGG
from mejiro.instruments.roman import Roman
from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.analysis.lens_subtraction import fit_sersic, subtract_lens


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(mejiro.__file__)), 'tests', 'test_data')


def test_fit_sersic_roundtrip():
    """
    Test that fit_sersic recovers known Sersic parameters from a synthetic image.
    """
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Util import util as lenstronomy_util

    # generate a known Sersic image
    num_pix = 51
    pixel_scale = 0.11  # Roman-like
    true_kwargs = {
        'amp': 500.0,
        'R_sersic': 0.4,
        'n_sersic': 4.0,
        'center_x': 0.05,
        'center_y': -0.03,
        'e1': 0.02,
        'e2': 0.04,
    }

    x, y, _, _, _, _, _, _ = lenstronomy_util.make_grid_with_coordtransform(
        numPix=num_pix, deltapix=pixel_scale, subgrid_res=1,
        left_lower=False, inverse=False)
    light_model = LightModel(['SERSIC_ELLIPSE'])
    true_image = light_model.surface_brightness(x, y, [true_kwargs]).reshape(num_pix, num_pix)

    # fit it
    best_fit, model_image, result = fit_sersic(true_image, pixel_scale)

    assert result.success, f'Optimization did not converge: {result.message}'

    # check parameter recovery
    assert abs(best_fit['R_sersic'] - true_kwargs['R_sersic']) / true_kwargs['R_sersic'] < 0.1, \
        f"R_sersic: expected {true_kwargs['R_sersic']}, got {best_fit['R_sersic']:.4f}"
    assert abs(best_fit['n_sersic'] - true_kwargs['n_sersic']) / true_kwargs['n_sersic'] < 0.1, \
        f"n_sersic: expected {true_kwargs['n_sersic']}, got {best_fit['n_sersic']:.4f}"
    assert abs(best_fit['center_x'] - true_kwargs['center_x']) < 0.05, \
        f"center_x: expected {true_kwargs['center_x']}, got {best_fit['center_x']:.4f}"
    assert abs(best_fit['center_y'] - true_kwargs['center_y']) < 0.05, \
        f"center_y: expected {true_kwargs['center_y']}, got {best_fit['center_y']:.4f}"

    # check that the residual is small
    residual = true_image - model_image
    relative_residual = np.sum(residual ** 2) / np.sum(true_image ** 2)
    assert relative_residual < 1e-4, \
        f'Relative residual {relative_residual:.6e} too large'


def test_subtract_lens_on_exposure():
    """
    Test subtract_lens on a noiseless Exposure with SampleGG.

    After subtracting the lens, the residual should contain mostly the lensed
    source arcs and be significantly fainter than the original.
    """
    strong_lens = SampleGG()
    band = 'F129'
    detector = 'SCA01'
    detector_position = (2048, 2048)

    kwargs_psf = STPSFEngine.get_roman_psf_kwargs(
        band, detector, detector_position, oversample=5, num_pix=101,
        check_cache=True, psf_cache_dir=TEST_DATA_DIR)

    synthetic_image = SyntheticImage(
        strong_lens=strong_lens,
        instrument=Roman(),
        band=band,
        fov_arcsec=5,
        instrument_params={'detector': detector, 'detector_position': detector_position},
        kwargs_numerics={},
        kwargs_psf=kwargs_psf,
        pieces=True)

    engine_params = {
        'sky_background': False,
        'detector_effects': False,
    }
    exposure = Exposure(
        synthetic_image,
        exposure_time=146,
        engine='galsim',
        engine_params=engine_params)

    residual, model, best_fit, fit_result = subtract_lens(exposure)

    # model should be non-negative and have meaningful flux
    assert np.sum(model) > 0, 'Model should have positive total flux'

    # residual should have less total flux than the original
    # (the lens is the dominant light source in the center)
    original_peak = np.max(exposure.exposure)
    residual_peak = np.max(np.abs(residual))
    assert residual_peak < original_peak, \
        f'Residual peak ({residual_peak:.1f}) should be smaller than original peak ({original_peak:.1f})'

    # the central region (lens-dominated) should be mostly subtracted
    center = exposure.exposure.shape[0] // 2
    half_width = 5  # pixels
    central_original = np.sum(exposure.exposure[center-half_width:center+half_width,
                                                 center-half_width:center+half_width])
    central_residual = np.sum(np.abs(residual[center-half_width:center+half_width,
                                               center-half_width:center+half_width]))
    assert central_residual < central_original, \
        'Central region should have less flux after lens subtraction'
