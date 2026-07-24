import os

import pytest
import galsim
import numpy as np

from mejiro.exposure import Exposure, LightweightExposure
from mejiro.synthetic_image import SyntheticImage
from mejiro.galaxy_galaxy import Sample1, SampleGG
from mejiro.instruments.roman import Roman
from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.utils import util


@pytest.mark.parametrize("strong_lens", [SampleGG(), Sample1()])
def test_exposure_with_galsim_engine(strong_lens, test_data_dir):
    from mejiro.engines.galsim_engine import GalSimEngine

    band = 'F129'
    detector = 'SCA01'
    detector_position = (2048, 2048)

    kwargs_psf = STPSFEngine.get_roman_psf_kwargs(band, detector, detector_position, oversample=5, num_pix=101, check_cache=True, psf_cache_dir=test_data_dir)

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


def test_noiseless_exposure_aperture_photometry(sample_gg, test_data_dir):
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
        check_cache=True, psf_cache_dir=test_data_dir)

    synthetic_image = SyntheticImage(
        strong_lens=sample_gg,
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


def _build_small_exposure(pieces):
    """Small, fast, deterministic Roman/F129 galsim exposure for lightweight tests.

    Uses ``kwargs_psf={}`` (no PSF-cache dependency) and disables sky background
    and detector effects, so ``exposure.data`` is deterministic. ``SampleGG`` is
    fine here because ``Exposure.save_lightweight`` only reads ``name``/``z_lens``/
    ``z_source`` off the lens (not the ``get_*`` accessors the SyntheticImage
    lightweight serializer needs).
    """
    synthetic_image = SyntheticImage(
        strong_lens=SampleGG(),
        instrument=Roman(),
        band='F129',
        fov_arcsec=2,
        instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
        kwargs_numerics={},
        kwargs_psf={},
        pieces=pieces,
    )
    return Exposure(
        synthetic_image,
        exposure_time=146,
        engine='galsim',
        engine_params={'sky_background': False, 'detector_effects': False},
    )


@pytest.mark.parametrize('pieces', [False, True])
def test_save_lightweight_roundtrip(tmp_path, pieces):
    """Exposure.save_lightweight + util.load_exposure preserves the data array,
    the pieces (iff pieces=True), and the scalar metadata consumers read."""
    orig = _build_small_exposure(pieces)
    path = str(tmp_path / 'roundtrip.npz')
    orig.save_lightweight(path)

    loaded = util.load_exposure(path)
    assert isinstance(loaded, LightweightExposure)

    np.testing.assert_array_equal(loaded.data, orig.data.astype(np.float32))
    assert loaded.exposure_time == orig.exposure_time
    assert loaded.engine == orig.engine
    assert loaded.band == orig.synthetic_image.band
    assert loaded.instrument_name == orig.synthetic_image.instrument_name
    assert loaded.num_pix == orig.synthetic_image.num_pix
    assert loaded.pixel_scale == pytest.approx(orig.synthetic_image.pixel_scale)
    assert loaded.name == orig.synthetic_image.strong_lens.name
    assert loaded.z_lens == pytest.approx(orig.synthetic_image.strong_lens.z_lens)
    assert loaded.z_source == pytest.approx(orig.synthetic_image.strong_lens.z_source)
    assert loaded.synthetic_image is None

    if pieces:
        np.testing.assert_array_equal(loaded.lens_data, orig.lens_data.astype(np.float32))
        np.testing.assert_array_equal(loaded.source_data, orig.source_data.astype(np.float32))
    else:
        assert loaded.lens_data is None
        assert loaded.source_data is None


def test_lightweight_exposure_get_snr(tmp_path):
    """get_snr works on a loaded lightweight exposure when pieces were saved,
    and raises the pieces-required ValueError when they were not."""
    # with pieces: get_snr resolves to a real, positive SNR
    orig = _build_small_exposure(pieces=True)
    path = str(tmp_path / 'pieces.npz')
    orig.save_lightweight(path)
    loaded = util.load_exposure(path)
    snr = loaded.get_snr()
    assert snr is not None and snr > 0

    # without pieces: SNR needs lens/source data -> ValueError (same as full Exposure)
    orig_np = _build_small_exposure(pieces=False)
    path_np = str(tmp_path / 'nopieces.npz')
    orig_np.save_lightweight(path_np)
    loaded_np = util.load_exposure(path_np)
    with pytest.raises(ValueError):
        loaded_np.get_snr()


def test_lightweight_file_size(tmp_path):
    """Guard against regressions that let the heavy galsim Image objects leak
    back into the .npz: a small exposure (with pieces) fits easily under 200 KB."""
    orig = _build_small_exposure(pieces=True)
    path = str(tmp_path / 'size.npz')
    orig.save_lightweight(path)
    size_bytes = os.path.getsize(path)
    assert size_bytes < 200_000, (
        f"lightweight .npz unexpectedly large ({size_bytes} bytes); "
        f"likely the galsim Image objects or noise arrays leaked in"
    )


def test_full_pickle_roundtrip_via_load_exposure(tmp_path):
    """The 'full' serialization branch (imaging.serialization: full) pickles the
    whole Exposure; util.load_exposure must round-trip it back with .data intact.
    Covers the .pkl path since every shipped config now defaults to lightweight."""
    orig = _build_small_exposure(pieces=True)
    path = str(tmp_path / 'full.pkl')
    util.pickle(path, orig)

    loaded = util.load_exposure(path)
    assert isinstance(loaded, Exposure)
    np.testing.assert_array_equal(loaded.data, orig.data)
    np.testing.assert_array_equal(loaded.source_data, orig.source_data)
    assert loaded.exposure_time == orig.exposure_time
    assert loaded.synthetic_image is None  # dropped by Exposure.__getstate__ on pickle


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
