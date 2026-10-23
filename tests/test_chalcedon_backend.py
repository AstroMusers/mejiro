import builtins
import numpy as np
import pytest
from astropy.cosmology import Planck18
import astropy.units as u


def test_strong_lens_simulation_does_not_import_lenstronomy(monkeypatch):
    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == 'lenstronomy' or name.startswith('lenstronomy.'):
            raise AssertionError('Native simulation must not import Lenstronomy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', block_import)
    from mejiro.galaxy_galaxy import GalaxyGalaxy
    from mejiro.synthetic_image import SyntheticImage

    class Detector:
        name = 'Native test detector'
        bands = ['test']

        def default_params(self):
            return {}

        def get_pixel_scale(self, band):
            return 0.05 * u.arcsec / u.pixel

        def get_zeropoint_magnitude(self, band):
            return 25.0

    lens = GalaxyGalaxy(
        'native-sis', None,
        {'lens_model_list': ['SIS'], 'lens_light_model_list': ['SERSIC_ELLIPSE'],
         'source_light_model_list': ['SERSIC_ELLIPSE'], 'lens_redshift_list': [0.5],
         'source_redshift_list': [1.5], 'z_source': 1.5, 'cosmo': Planck18},
        {'kwargs_lens': [{'theta_E': 1.0}],
         'kwargs_lens_light': [{'amp': 0.1, 'R_sersic': 0.3, 'n_sersic': 1.0, 'e1': 0.0, 'e2': 0.0}],
         'kwargs_source': [{'amp': 1.0, 'R_sersic': 0.05, 'n_sersic': 1.0,
                            'center_x': 0.1, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0}],
         'kwargs_ps': []},
    )
    image = SyntheticImage(lens, Detector(), 'test', fov_arcsec=4.0, pieces=True)

    assert image.data.shape == (81, 81)
    assert np.isfinite(image.data).all()
    assert image.source_surface_brightness.max() > 0
    np.testing.assert_allclose(image.data, image.source_surface_brightness + image.lens_surface_brightness)
    positions = lens.get_image_positions()
    np.testing.assert_allclose(np.sort(positions[0]), [-0.9, 1.1], atol=1e-6)
    assert lens.get_lens_cosmo().sis_theta_E2sigma_v(1.0) > 0


def test_builtin_analytic_sample_uses_native_model(monkeypatch):
    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == 'lenstronomy' or name.startswith('lenstronomy.'):
            raise AssertionError('Native sample must not import Lenstronomy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', block_import)
    from mejiro.galaxy_galaxy import SampleGG

    lens = SampleGG()
    source_x, source_y = lens.lens_model.ray_shooting([1.0], [0.2], lens.kwargs_lens)
    assert np.isfinite(source_x).all() and np.isfinite(source_y).all()


def test_roman_sample_renders_without_lenstronomy_and_preserves_reference_images(monkeypatch):
    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == 'lenstronomy' or name.startswith('lenstronomy.'):
            raise AssertionError('Roman simulation must not import Lenstronomy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', block_import)
    from mejiro.galaxy_galaxy import SampleGG
    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage

    lens = SampleGG()
    image = SyntheticImage(lens, Roman(), 'F129', fov_arcsec=4.0, pieces=True)
    positions = lens.get_image_positions()

    assert np.isfinite(image.data).all()
    assert image.data.sum() > 0
    zeropoint = getattr(image.magnitude_zeropoint, 'value', image.magnitude_zeropoint)  # [mag]
    expected_maggies = float(image.data.sum()) * 10.0**(-0.4 * zeropoint)
    assert image.get_maggies() == pytest.approx(expected_maggies, rel=1e-13)
    np.testing.assert_allclose(image.data, image.lens_surface_brightness + image.source_surface_brightness)
    np.testing.assert_allclose(positions[0], [1.12731457, -0.56841653], atol=1e-6)
    np.testing.assert_allclose(positions[1], [-1.50967129, 0.57814324], atol=1e-6)


def test_jwst_and_lsst_instrument_defaults_do_not_import_lenstronomy(monkeypatch):
    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == 'lenstronomy' or name.startswith('lenstronomy.'):
            raise AssertionError('Instrument configuration must not import Lenstronomy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', block_import)
    from mejiro.instruments.jwst import JWST
    from mejiro.instruments.lsst import LSST

    jwst = JWST()
    lsst = LSST()

    assert jwst.get_zeropoint_magnitude('F115W') == 28.02
    assert lsst.get_zeropoint_magnitude('r') == 28.13
    assert jwst.get_pixel_scale('F115W').value == 0.031
    assert lsst.get_psf_fwhm('r').value == 0.73


def test_native_lens_light_subtraction_requires_no_external_lensing_engine(monkeypatch):
    from chalcedon import LightModel
    from chalcedon.imaging import make_grid_with_coordtransform
    from mejiro.analysis.lens_subtraction import fit_sersic

    parameters = {'amp': 2.0, 'R_sersic': 0.3, 'n_sersic': 1.0,
                  'center_x': 0.0, 'center_y': 0.0, 'e1': 0.1, 'e2': 0.0}
    coordinates = make_grid_with_coordtransform(31, 0.05)
    image = LightModel(['SERSIC_ELLIPSE']).surface_brightness(coordinates[0], coordinates[1], [parameters]).reshape(31, 31)

    recovered, fitted, result = fit_sersic(image, 0.05, initial_kwargs=parameters)

    np.testing.assert_allclose(fitted, image, atol=1e-7)
    assert recovered['R_sersic'] == pytest.approx(0.3, rel=1e-4)


def test_native_example_renders_without_lenstronomy(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    import matplotlib.image as mpimg

    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == 'lenstronomy' or name.startswith('lenstronomy.'):
            raise AssertionError('Native example must not import Lenstronomy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', block_import)
    path = Path(__file__).parents[1] / 'examples' / 'native_strong_lens.py'
    specification = importlib.util.spec_from_file_location('native_strong_lens_example', path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    output_path = tmp_path / 'native_strong_lens.png'
    image = example.run_example(output_path)

    assert np.isfinite(image.data).all()
    assert image.source_surface_brightness.sum() > 0
    assert mpimg.imread(output_path)[..., :3].min() < 0.8


@pytest.mark.parametrize('sample_name', ['Sample1', 'Sample2', 'SampleSL2S', 'SampleBELLS'])
def test_builtin_halo_and_shapelet_samples_render_with_native_backend(monkeypatch, sample_name):
    original_import = builtins.__import__

    def block_import(name, *args, **kwargs):
        if name == 'lenstronomy' or name.startswith('lenstronomy.'):
            raise AssertionError('Built-in scene must not import Lenstronomy')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', block_import)
    from mejiro import galaxy_galaxy
    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage

    lens = getattr(galaxy_galaxy, sample_name)()
    image = SyntheticImage(lens, Roman(), 'F129', fov_arcsec=3.0, pieces=True)

    assert np.isfinite(image.data).all()
    assert image.source_surface_brightness.max() > 0
    np.testing.assert_allclose(image.data, image.lens_surface_brightness + image.source_surface_brightness)