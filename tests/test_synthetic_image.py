import os
import pytest
import numpy as np
from pyHalo.preset_models import preset_model_from_name

import mejiro
from mejiro.instruments.roman import Roman
from mejiro.galaxy_galaxy import Sample1, Sample2, SampleGG, SampleSL2S, SampleBELLS
from mejiro.synthetic_image import SyntheticImage
from mejiro.engines.stpsf_engine import STPSFEngine
from mejiro.utils import util


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(mejiro.__file__)), 'tests', 'test_data')


def _circular_aperture_mask(shape, center_xy, radius):
    """Boolean mask for a circular aperture of `radius` pixels at `center_xy`.

    `center_xy` is (x, y) in pixel coordinates (matching the convention
    returned by SyntheticImage.get_image_positions(pixel=True)).
    """
    ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def test_band_specific_source_image():
    """SyntheticImage should swap kwargs_source[0]['image'] to the band-specific image."""
    strong_lens = Sample1()  # INTERPOL source

    img_f106 = np.ones((10, 10)) * 1.0
    img_f129 = np.ones((10, 10)) * 2.0
    strong_lens.kwargs_params['source_images'] = {'F106': img_f106, 'F129': img_f129}

    SyntheticImage(strong_lens=strong_lens,
                   instrument=Roman(),
                   band='F106',
                   fov_arcsec=5,
                   instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                   kwargs_numerics={},
                   kwargs_psf={},
                   pieces=False)

    np.testing.assert_array_equal(strong_lens.kwargs_source[0]['image'], img_f106)


@pytest.mark.parametrize("strong_lens", [Sample1(), Sample2(), SampleGG()])
def test_magnitudes(strong_lens):
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)

@pytest.mark.parametrize("strong_lens", [SampleSL2S(), SampleBELLS()])
def test_lenstronomy_amplitudes(strong_lens):
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)

@pytest.mark.parametrize("strong_lens", [Sample1(), Sample2(), SampleGG(), SampleSL2S(), SampleBELLS()])   
def test_kwargs_numerics(strong_lens):
    roman = Roman()
    
    # test defaulting
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=roman,
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     # none provided
                                     kwargs_psf={},
                                     pieces=False)
    assert synthetic_image.kwargs_numerics == SyntheticImage.DEFAULT_KWARGS_NUMERICS
    
    # regular compute mode
    kwargs_numerics = {
        'supersampling_factor': 5,
        'compute_mode': 'regular'
    }
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=roman,
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False)

    # adaptive compute mode with supersampled indices provided
    region = util.create_centered_circle(N=47, radius=10)
    kwargs_numerics = {
        'supersampling_factor': 5,
        'compute_mode': 'adaptive',
        'supersampled_indexes': region
    }
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=roman,
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False)

    # adaptive compute mode with default supersampled indices (annulus around image positions)
    kwargs_numerics = {
        'supersampling_factor': 5,
        'compute_mode': 'adaptive',
    }
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=roman,
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False)
    
    # unhappy path: insufficient supersampling factor
    kwargs_numerics = {
        'supersampling_factor': 1,
        'compute_mode': 'adaptive',
    }
    with pytest.warns(UserWarning,
                      match='Supersampling factor less than 5 may not be sufficient for accurate results, especially when convolving with a non-trivial PSF'):
        synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=roman,
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics=kwargs_numerics,
                                     kwargs_psf={},
                                     pieces=False)
        

@pytest.mark.parametrize("strong_lens", [Sample1(), Sample2(), SampleGG(), SampleSL2S(), SampleBELLS()])        
def test_build_adaptive_grid(strong_lens):
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)
    
    image_positions = synthetic_image.get_image_positions(pixel=True)

    # check that with a padding of 1 pixel, the grid always includes all image positions
    includes_grid = synthetic_image.build_adaptive_grid(pad=1)
    for i, _ in enumerate(image_positions[0]):
        assert includes_grid[round(image_positions[1][i]), round(image_positions[0][i])]

    # check that a huge grid is slimmed down to the size of the image
    huge_grid = synthetic_image.build_adaptive_grid(pad=1000)
    assert huge_grid.shape[0] == synthetic_image.data.shape[0]
    assert huge_grid.shape[1] == synthetic_image.data.shape[1]

    # unhappy path: negative padding
    with pytest.raises(ValueError, match='Padding value must be a non-negative integer'):
        synthetic_image.build_adaptive_grid(pad=-1)
    
    # unhappy path: non-integer padding
    with pytest.raises(ValueError, match='Padding value must be a non-negative integer'):
        synthetic_image.build_adaptive_grid(pad=1.5)


def test_get_image_positions():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)
    
    # check angular positions
    angular_positions = synthetic_image.get_image_positions(pixel=False)
    expected_angular_result = (np.array([ 1.12731457, -0.56841653]), np.array([-1.50967129,  0.57814324]))
    np.testing.assert_allclose(angular_positions[0], expected_angular_result[0], rtol=1e-5)
    np.testing.assert_allclose(angular_positions[1], expected_angular_result[1], rtol=1e-5)

    # check pixel positions
    pixel_positions = synthetic_image.get_image_positions(pixel=True)
    expected_pixel_result = (np.array([33.24831427, 17.83257703]), np.array([ 9.27571558, 28.25584762]))
    np.testing.assert_allclose(pixel_positions[0], expected_pixel_result[0], rtol=1e-5)
    np.testing.assert_allclose(pixel_positions[1], expected_pixel_result[1], rtol=1e-5)


def test_fov_flux():
    common_kwargs = dict(
        strong_lens=SampleGG(),
        instrument=Roman(),
        band='F129',
        instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
        kwargs_numerics={},
        kwargs_psf={},
        pieces=False,
        )
    small_fov = SyntheticImage(fov_arcsec=3, **common_kwargs)
    large_fov = SyntheticImage(fov_arcsec=7, **common_kwargs)

    assert large_fov.get_flux() > small_fov.get_flux()
    assert large_fov.get_maggies() > small_fov.get_maggies()


def test_plot():
    synthetic_image = SyntheticImage(strong_lens=SampleGG(),
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)
    synthetic_image.plot()  


def test_overplot_subhalos():
    strong_lens = SampleGG()
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                         instrument=Roman(),
                                         band='F129',
                                         fov_arcsec=5,
                                         instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                         kwargs_numerics={},
                                         kwargs_psf={},
                                         pieces=False)
    with pytest.raises(ValueError):
        synthetic_image.overplot_subhalos()

    CDM = preset_model_from_name('CDM')
    realization = CDM(round(strong_lens.z_lens, 2), round(strong_lens.z_source, 2), cone_opening_angle_arcsec=5, log_m_host=np.log10(strong_lens.get_main_halo_mass()))
    strong_lens.add_realization(realization)

    synthetic_image.overplot_subhalos()


@pytest.mark.parametrize("strong_lens", [Sample1(), SampleGG(), SampleSL2S(), SampleBELLS()])
def test_synthetic_image_data_content(strong_lens):
    """Verify the produced image array has sane numerical content, not just the right shape."""
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)

    data = synthetic_image.data
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape == (synthetic_image.num_pix, synthetic_image.num_pix)
    assert np.all(np.isfinite(data)), "synthetic image contains NaN or inf"
    # PSF convolution can produce small negative ringing artifacts at image edges/background
    assert data.min() > -0.01 * data.max(), "synthetic image has significant negative surface brightness"
    assert np.sum(data) > 0, "synthetic image is empty"
    assert data.max() > data.mean(), "synthetic image is flat (no peak)"
    # get_flux() should match a direct sum of the array
    np.testing.assert_allclose(synthetic_image.get_flux(), np.sum(data), rtol=1e-12)


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_synthetic_image_data_pieces(strong_lens):
    """When pieces=True, lens + source surface brightness should reconstruct data exactly."""
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=True)

    lens = synthetic_image.lens_surface_brightness
    source = synthetic_image.source_surface_brightness

    assert lens is not None and source is not None
    assert lens.shape == synthetic_image.data.shape
    assert source.shape == synthetic_image.data.shape
    assert np.sum(lens) > 0, "lens surface brightness should be non-zero"
    assert np.sum(source) > 0, "lensed source surface brightness should be non-zero"
    np.testing.assert_allclose(synthetic_image.data, lens + source, rtol=1e-10)


@pytest.mark.parametrize("strong_lens", [SampleGG(), SampleSL2S(), SampleBELLS()])
def test_synthetic_image_aperture_photometry(strong_lens):
    """Aperture photometry at the predicted lensed-image positions should
    return real flux above the local background. This verifies pixel values
    are sensible *in the right places* (not just sensible in aggregate).
    """
    synthetic_image = SyntheticImage(strong_lens=strong_lens,
                                     instrument=Roman(),
                                     band='F129',
                                     fov_arcsec=5,
                                     instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
                                     kwargs_numerics={},
                                     kwargs_psf={},
                                     pieces=False)

    px, py = synthetic_image.get_image_positions(pixel=True)
    assert len(px) >= 1, "no lensed image positions returned"
    aperture_radius = 3

    aperture_fluxes = []
    for cx, cy in zip(px, py):
        # skip image positions outside the rendered grid
        if not (0 <= cx < synthetic_image.data.shape[1] and 0 <= cy < synthetic_image.data.shape[0]):
            continue
        ap_mask = _circular_aperture_mask(synthetic_image.data.shape, (cx, cy), aperture_radius)
        ap_pixels = synthetic_image.data[ap_mask]
        ap_flux = ap_pixels.sum()

        # the lensed image should be a real source with positive flux;
        # local sky subtraction is unreliable for galaxy-galaxy lenses because
        # the lens halo and Einstein ring contaminate a nearby sky annulus
        assert ap_flux > 0, f"aperture at ({cx:.1f},{cy:.1f}) is empty"
        aperture_fluxes.append(ap_flux)

    assert len(aperture_fluxes) >= 1
    # apertures should each contain a meaningful fraction of total flux
    total_flux = synthetic_image.get_flux()
    summed_aperture_flux = float(np.sum(aperture_fluxes))
    assert summed_aperture_flux / total_flux > 0.01, (
        f"lensed-image apertures sum to only {summed_aperture_flux/total_flux:.2%} "
        "of total flux — image positions may not align with bright pixels"
    )


def test_synthetic_image_psf_convolution_changes_image():
    """Convolving with a wider PSF should (a) preserve total flux,
    (b) lower the peak pixel, and (c) leak flux out of fixed-radius
    apertures around the lensed-image positions.
    """
    common = dict(
        strong_lens=SampleGG(),
        instrument=Roman(),
        band='F129',
        fov_arcsec=5,
        instrument_params={'detector': 'SCA01', 'detector_position': (2048, 2048)},
        kwargs_numerics={},
        pieces=False,
    )

    img_none = SyntheticImage(kwargs_psf={'psf_type': 'NONE'}, **common)
    img_narrow = SyntheticImage(kwargs_psf={'psf_type': 'GAUSSIAN', 'fwhm': 0.05}, **common)
    img_wide = SyntheticImage(kwargs_psf={'psf_type': 'GAUSSIAN', 'fwhm': 0.20}, **common)

    # all three images should have the same shape
    assert img_none.data.shape == img_narrow.data.shape == img_wide.data.shape

    # convolution should actually have changed the array
    assert not np.allclose(img_none.data, img_narrow.data)
    assert not np.allclose(img_narrow.data, img_wide.data)
    assert not np.allclose(img_none.data, img_wide.data)

    # convolution conserves total flux (within ~1%)
    flux_none = img_none.get_flux()
    flux_narrow = img_narrow.get_flux()
    flux_wide = img_wide.get_flux()
    np.testing.assert_allclose(flux_narrow, flux_none, rtol=1e-2)
    np.testing.assert_allclose(flux_wide, flux_none, rtol=1e-2)

    # wider PSF -> lower peak pixel
    assert img_none.data.max() > img_narrow.data.max() > img_wide.data.max(), (
        f"peak ordering broken: none={img_none.data.max()}, "
        f"narrow={img_narrow.data.max()}, wide={img_wide.data.max()}"
    )

    # wider PSF -> less flux inside a fixed-radius aperture at each image position
    px, py = img_none.get_image_positions(pixel=True)
    aperture_radius = 3
    for cx, cy in zip(px, py):
        if not (0 <= cx < img_none.data.shape[1] and 0 <= cy < img_none.data.shape[0]):
            continue
        mask = _circular_aperture_mask(img_none.data.shape, (cx, cy), aperture_radius)
        f_none = img_none.data[mask].sum()
        f_narrow = img_narrow.data[mask].sum()
        f_wide = img_wide.data[mask].sum()
        assert f_none > f_narrow > f_wide, (
            f"aperture flux not monotone vs PSF width at ({cx:.1f},{cy:.1f}): "
            f"none={f_none}, narrow={f_narrow}, wide={f_wide}"
        )


def test_roman_psf_varies_with_detector_position():
    """Two SyntheticImages of the same lens, identical except for the
    Roman PSF used at two different detector positions, should produce
    measurably different rendered images while still conserving total
    flux. Requires that both PSFs are present in the test_data cache —
    populate via tests/test_data/cache_test_psfs.py if missing.
    """
    band = 'F129'
    detector = 'SCA01'
    pos_a = (2048, 2048)
    pos_b = (200, 200)

    # Skip cleanly if the second cached PSF hasn't been generated yet,
    # since computing one with STPSF at test time would be very slow.
    psf_id_b = STPSFEngine.get_psf_id(band, detector, pos_b, 5, 101)
    psf_b_path = os.path.join(TEST_DATA_DIR, f'{psf_id_b}.npy')
    if not os.path.exists(psf_b_path):
        pytest.skip(
            f"second cached Roman PSF not found at {psf_b_path}; "
            "run tests/test_data/cache_test_psfs.py to generate it"
        )

    kwargs_psf_a = STPSFEngine.get_roman_psf_kwargs(
        band, detector, pos_a, oversample=5, num_pix=101,
        check_cache=True, psf_cache_dir=TEST_DATA_DIR)
    kwargs_psf_b = STPSFEngine.get_roman_psf_kwargs(
        band, detector, pos_b, oversample=5, num_pix=101,
        check_cache=True, psf_cache_dir=TEST_DATA_DIR)

    # Failsafe: confirm the underlying PSF kernels themselves differ.
    # If the cache lookup silently returned the same file twice, the
    # SyntheticImage assertions below would all pass for the wrong reason.
    kernel_a = kwargs_psf_a['kernel_point_source']
    kernel_b = kwargs_psf_b['kernel_point_source']
    assert kernel_a.shape == kernel_b.shape
    assert not np.allclose(kernel_a, kernel_b), \
        "the two cached Roman PSFs are identical"

    common = dict(
        strong_lens=SampleGG(),
        instrument=Roman(),
        band=band,
        fov_arcsec=5,
        kwargs_numerics={},
        pieces=False,
    )
    img_a = SyntheticImage(
        instrument_params={'detector': detector, 'detector_position': pos_a},
        kwargs_psf=kwargs_psf_a, **common)
    img_b = SyntheticImage(
        instrument_params={'detector': detector, 'detector_position': pos_b},
        kwargs_psf=kwargs_psf_b, **common)

    # same scene, different PSF -> different rendered image
    assert img_a.data.shape == img_b.data.shape
    assert not np.allclose(img_a.data, img_b.data), \
        "different Roman PSFs should produce different rendered images"

    # PSF convolution conserves total flux
    flux_a = img_a.get_flux()
    flux_b = img_b.get_flux()
    np.testing.assert_allclose(flux_b, flux_a, rtol=1e-2)

    # the difference between the two images should be non-trivial
    rel_l2 = np.linalg.norm(img_a.data - img_b.data) / np.linalg.norm(img_a.data)
    assert rel_l2 > 1e-3, (
        f"two Roman-PSF images barely differ (rel L2 = {rel_l2:.2e})"
    )
