import numpy as np
import pytest

from mejiro.lensing import get_backend, validate_backend


def test_both_backends_produce_the_analytic_sis_deflection():
    for backend in ('lenstronomy', 'chalcedon'):
        model = get_backend(backend).LensModel(['SIS'])
        alpha = model.alpha(np.array([1.0, -2.0]), np.array([0.0, 0.0]), [{'theta_E': 0.5}])
        np.testing.assert_allclose(alpha, [[0.5, -0.5], [0.0, 0.0]])


def test_backend_names_are_explicit_and_invalid_choices_raise():
    assert validate_backend('lenstronomy') == 'lenstronomy'
    assert validate_backend('chalcedon') == 'chalcedon'
    with pytest.raises(ValueError, match='backend'):
        get_backend('unknown')


@pytest.mark.parametrize('name', ['', None, False, {}])
def test_invalid_backend_types_and_empty_name_are_rejected(name):
    with pytest.raises(ValueError, match='backend'):
        validate_backend(name)


@pytest.mark.parametrize('backend', ['lenstronomy', 'chalcedon'])
def test_same_lens_renders_with_each_selected_backend_without_mutating_original(backend):
    from mejiro.galaxy_galaxy import SampleGG
    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage

    lens = SampleGG(backend='lenstronomy')
    cached_cosmology = lens.get_lens_cosmo()
    image = SyntheticImage(lens, Roman(), 'F129', fov_arcsec=3.0, backend=backend, pieces=True)

    assert image.backend == backend
    assert image.strong_lens.backend == backend
    assert lens.backend == 'lenstronomy'
    assert lens.get_lens_cosmo() is cached_cosmology
    assert np.isfinite(image.data).all() and image.data.sum() > 0
    np.testing.assert_allclose(image.data, image.lens_surface_brightness + image.source_surface_brightness)
    expected_type = get_backend(backend).LensModel
    assert isinstance(image.strong_lens.lens_model, expected_type)


def test_default_rendering_uses_chalcedon_backend():
    from mejiro.galaxy_galaxy import SampleGG
    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage

    lens = SampleGG()
    image = SyntheticImage(lens, Roman(), 'F129', fov_arcsec=2.0)

    assert lens.backend == image.backend == 'chalcedon'
    assert image.strong_lens is lens


def test_constructor_selects_chalcedon_for_models_cosmology_and_positions():
    from mejiro.galaxy_galaxy import SampleGG

    lens = SampleGG(backend='chalcedon')
    selected = get_backend('chalcedon')

    assert isinstance(lens.lens_model, selected.LensModel)
    assert isinstance(lens.lens_light_model, selected.LightModel)
    assert isinstance(lens.source_light_model, selected.LightModel)
    assert isinstance(lens.get_lens_cosmo(), selected.LensCosmo)
    positions = lens.get_image_positions()
    np.testing.assert_allclose(positions[0], [1.12731457, -0.56841653], atol=1e-6)


@pytest.mark.parametrize('backend', ['lenstronomy', 'chalcedon'])
def test_lightweight_images_preserve_backend_metadata(backend, tmp_path):
    from mejiro.galaxy_galaxy import Sample1
    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage, LightweightSyntheticImage

    image = SyntheticImage(Sample1(backend=backend), Roman(), 'F129', fov_arcsec=2.0)
    path = tmp_path / 'scene.npz'
    image.save_lightweight(str(path))
    restored = LightweightSyntheticImage.load(path)

    assert restored.backend == backend
    np.testing.assert_allclose(restored.data, image.data, rtol=1e-6)


def test_old_lightweight_image_metadata_remains_readable():
    from mejiro.synthetic_image import LightweightSyntheticImage

    metadata = {'schema_version': 2, 'band': 'test', 'pixel_scale': 0.1,
                'fov_arcsec': 0.1, 'num_pix': 1, 'instrument_name': 'test',
                'magnitude_zeropoint': 25.0,
                'lens': {'name': 'old-scene', 'z_lens': 0.5, 'z_source': 1.5,
                         'has_realization': False, 'main_halo_mass': 1e12,
                         'einstein_radius': 1.0, 'velocity_dispersion': 220.0,
                         'magnification': 2.0, 'lens_magnitude': 21.0,
                         'source_magnitude': 24.0, 'lensed_source_magnitude': 23.0}}
    image = LightweightSyntheticImage(np.ones((1, 1)), metadata)

    assert image.backend == 'unknown'