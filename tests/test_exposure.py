import os
import pytest
import galsim
import numpy as np
import os
import pytest

import mejiro
from mejiro.exposure import Exposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.galaxy_galaxy import Sample1, Sample2, SampleGG, SampleSL2S, SampleBELLS
from mejiro.instruments.roman import Roman
from mejiro.engines.stpsf_engine import STPSFEngine


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(mejiro.__file__)), 'tests', 'test_data')


@pytest.mark.parametrize("strong_lens", [SampleGG(), Sample1()])
def test_exposure_with_galsim_engine(strong_lens):
    from mejiro.engines.galsim_engine import GalSimEngine

    band = 'F129'
    detector = 'SCA01'
    detector_position = (2048, 2048)

    kwargs_psf = STPSFEngine.get_roman_psf_kwargs(band, detector, detector_position, oversample=5, num_pix=101, check_cache=True, psf_cache_dir=TEST_DATA_DIR)

    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band=band,
                                     fov_arcsec=5,
                                     instrument_params={'detector': detector, 'detector_position': detector_position},
                                     kwargs_numerics={},
                                     kwargs_psf=kwargs_psf,
                                     pieces=False)

    exposure = Exposure(synthetic_image,
                        exposure_time=146,
                        engine='galsim')

    assert exposure.synthetic_image == synthetic_image
    assert exposure.exposure_time == 146
    assert exposure.engine == 'galsim'

    # check engine param defaulting
    ignored_keys = ['rng']
    for key, item in exposure.engine_params.items():
        if key not in ignored_keys:
            assert item == GalSimEngine.default_roman_engine_params()[key]

    # noise
    assert exposure.noise is not None
    assert type(exposure.noise) is np.ndarray
    assert exposure.noise.shape == (synthetic_image.num_pix, synthetic_image.num_pix)
    # engine-specific noise components are tested in the engine-specific tests

    # exposure
    assert type(exposure.data) is np.ndarray
    assert exposure.data.shape == (synthetic_image.num_pix, synthetic_image.num_pix)

    # exposure data content (not just shape/type)
    assert np.all(np.isfinite(exposure.data)), "exposure.data contains NaN or inf"
    assert np.sum(exposure.data) > 0, "exposure.data is empty"
    assert exposure.data.max() > np.median(exposure.data), \
        "exposure.data has no peak above the median (background)"

    # image
    assert type(exposure.image) is galsim.Image
    assert exposure.image.array.shape == (synthetic_image.num_pix, synthetic_image.num_pix)
    np.testing.assert_array_equal(exposure.data, exposure.image.array)


def _circular_aperture_mask(shape, center_xy, radius):
    ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def test_noiseless_exposure_aperture_photometry():
    """A noiseless galsim exposure should preserve flux locally at each
    lensed-image position, not just globally. The aperture flux in the
    exposure should match the aperture flux in the synthetic image
    (scaled by exposure_time) within quantization error.
    """
    band = 'F129'
    detector = 'SCA01'
    detector_position = (2048, 2048)
    exposure_time = 146

    kwargs_psf = STPSFEngine.get_roman_psf_kwargs(
        band, detector, detector_position, oversample=5, num_pix=101,
        check_cache=True, psf_cache_dir=TEST_DATA_DIR)

    synthetic_image = SyntheticImage(
        strong_lens=SampleGG(),
        instrument=Roman(),
        band=band,
        fov_arcsec=5,
        instrument_params={'detector': detector, 'detector_position': detector_position},
        kwargs_numerics={},
        kwargs_psf=kwargs_psf,
        pieces=False,
    )

    # disable sky background and all detector effects -> exposure ~= data * exptime
    exposure = Exposure(
        synthetic_image,
        exposure_time=exposure_time,
        engine='galsim',
        engine_params={'sky_background': False, 'detector_effects': False},
    )

    px, py = synthetic_image.get_image_positions(pixel=True)
    aperture_radius = 3
    checked = 0
    for cx, cy in zip(px, py):
        if not (0 <= cx < synthetic_image.data.shape[1] and 0 <= cy < synthetic_image.data.shape[0]):
            continue
        mask = _circular_aperture_mask(synthetic_image.data.shape, (cx, cy), aperture_radius)
        synth_aperture_counts = synthetic_image.data[mask].sum() * exposure_time
        exposure_aperture_counts = exposure.data[mask].sum()

        # both apertures must contain a real source
        assert synth_aperture_counts > 0
        assert exposure_aperture_counts > 0

        # local flux should be preserved within quantization error.
        # Quantization error is at most ~0.5 per pixel; allow a few pixels of slack.
        n_pix = int(mask.sum())
        max_quant_err = max(0.5 * n_pix, 1.0)
        assert abs(exposure_aperture_counts - synth_aperture_counts) <= max_quant_err + 1e-6, (
            f"local flux not preserved at ({cx:.1f},{cy:.1f}): "
            f"synth={synth_aperture_counts:.2f}, exposure={exposure_aperture_counts:.2f}, "
            f"max_err={max_quant_err}"
        )
        checked += 1

    assert checked >= 1, "no aperture positions were inside the rendered grid"

# TODO awaiting fix of lenstronomy engine
# def test_exposure_with_lenstronomy_engine():
#     from mejiro.engines import lenstronomy_engine

#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     exposure = Exposure(synthetic_image,
#                         exposure_time=146,
#                         engine='lenstronomy',
#                         # default engine params
#                         check_cache=True,
#                         psf_cache_dir=TEST_DATA_DIR)

#     assert exposure.synthetic_image == synthetic_image
#     assert exposure.exposure_time == 146
#     assert exposure.engine == 'lenstronomy'

#     # check engine param defaulting
#     assert exposure.engine_params == lenstronomy_engine.default_roman_engine_params()

#     # noise
#     assert exposure.noise is not None
#     assert type(exposure.noise) is np.ndarray
#     assert exposure.noise.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)
#     # engine-specific noise components are tested in the engine-specific tests

#     # exposure
#     assert type(exposure.data) is np.ndarray
#     assert exposure.data.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)

# TODO need to fix Pandeia engine for new version
# def test_exposure_with_pandeia_engine():
#     from mejiro.engines import pandeia_engine

#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     # for Pandeia engine, default of 10^4 takes almost 10 minutes to run, so reduce number of samples
#     engine_params = pandeia_engine.default_roman_engine_params()
#     engine_params['num_samples'] = 10

#     exposure = Exposure(synthetic_image,
#                         exposure_time=146,
#                         engine='pandeia',
#                         engine_params=engine_params,
#                         check_cache=True,
#                         psf_cache_dir=TEST_DATA_DIR)

#     assert exposure.synthetic_image == synthetic_image
#     assert exposure.exposure_time == 146
#     assert exposure.engine == 'pandeia'

#     # check engine param defaulting
#     assert exposure.engine_params == engine_params

#     # noise
#     assert exposure.noise is not None
#     assert type(exposure.noise) is np.ndarray
#     assert exposure.noise.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)
#     # engine-specific noise components are tested in the engine-specific tests

#     # exposure
#     assert type(exposure.data) is np.ndarray
#     assert exposure.data.shape == (synthetic_image.native_num_pix, synthetic_image.native_num_pix)

# TODO TEMP
# def test_default_engine():
#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     exposure = Exposure(synthetic_image,
#                         exposure_time=146,
#                         # don't provide engine
#                         check_cache=True,
#                         psf_cache_dir=TEST_DATA_DIR)

#     assert exposure.engine == 'galsim'


# def test_invalid_engine():
#     synthetic_image = util.unpickle(f'{TEST_DATA_DIR}/synthetic_image_roman_F129_5_5.pkl')

#     try:
#         Exposure(synthetic_image,
#                  exposure_time=146,
#                  engine='invalid_engine',
#                  check_cache=True,
#                  psf_cache_dir=TEST_DATA_DIR)
#     except ValueError as e:
#         assert str(e) == 'Engine "invalid_engine" not recognized'


# def test_crop_edge_effects():
#     # unhappy path
#     image = np.zeros((100, 100))
#     with pytest.raises(AssertionError):
#         Exposure.crop_edge_effects(image)

#     # happy path
#     image = np.zeros((101, 101))
#     expected = np.zeros((98, 98))
#     result = Exposure.crop_edge_effects(image)
#     assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"
