from importlib import import_module

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
import numpy as np
from PIL import Image
import pytest

from tdpy.plotting import plot_background_colors, plot_style


@pytest.mark.parametrize('kind', ['percentile', 'log10', 'arcsinh', 'plot',
                                  'rgb_plot_grid', 'plot_grid', 'plot_list', 'log10_list'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_image_display_apis_honor_theme_without_changing_pixels(tmp_path, monkeypatch, kind, typeplotback, inherit):
    plotting = import_module('mejiro.plots.plot')
    image = np.arange(1.0, 17.0).reshape(4, 4)
    original = image.copy()
    images = [image * factor for factor in (1, 2, 3, 4)]
    background, foreground = plot_background_colors(typeplotback)
    displayed = []
    path = tmp_path / 'grid.png'

    def inspect_display():
        figure = plt.gcf()
        displayed.append(figure)
        assert figure.get_facecolor() == to_rgba(background)
        for axis in figure.axes:
            assert axis.get_facecolor() == to_rgba(background)
            assert axis.xaxis.label.get_color() == foreground
            assert axis.yaxis.label.get_color() == foreground
        FigureCanvasAgg(figure).draw()
        np.testing.assert_array_equal(np.asarray(figure.canvas.buffer_rgba())[0, 0],
                                      np.array(to_rgba(background)) * 255)

    monkeypatch.setattr(plt, 'show', inspect_display)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        plt.rcParams.update({'text.usetex': False, 'figure.dpi': 40, 'savefig.dpi': 40})
        original_settings = dict(plt.rcParams)
        try:
            if kind in {'percentile', 'log10', 'arcsinh', 'plot'}:
                assert getattr(plotting, kind)(image, colorbar=True, **options) is None
                if kind == 'log10':
                    expected = [np.log10(image / image.max())]
                elif kind == 'arcsinh':
                    expected = [plotting.asinh(image)]
                else:
                    expected = [image]
            elif kind == 'rgb_plot_grid':
                rgb = [np.repeat((array / array.max())[..., None], 3, axis=2) for array in images]
                assert plotting.rgb_plot_grid(rgb, 2, save=path, **options) is None
                expected = rgb
            elif kind == 'plot_grid':
                assert plotting.plot_grid(images, 2, save=path, colorbar=True, **options) is None
                expected = [np.log10(array) for array in images]
            else:
                assert getattr(plotting, kind)(images[:2], **options) is None
                expected = [np.log10(array) for array in images[:2]] if kind == 'log10_list' else images[:2]
            assert len(displayed) == 1
            figure = displayed[0]
            plotted = [artist.get_array() for axis in figure.axes for artist in axis.images]
            assert len(plotted) == len(expected)
            for values, reference in zip(plotted, expected):
                np.testing.assert_array_equal(values, reference)
            np.testing.assert_array_equal(image, original)
            assert dict(plt.rcParams) == original_settings
            if kind in {'rgb_plot_grid', 'plot_grid'}:
                with Image.open(path) as saved:
                    expected_background = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
                    assert saved.convert('RGB').getpixel((0, 0)) == expected_background
        finally:
            plt.close('all')


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_corner_plots_theme_returned_figures_and_preserve_population(typeplotback):
    from types import SimpleNamespace

    plotting = import_module('mejiro.plots.corner')
    data = np.random.default_rng(42).normal(size=(200, 7))
    lenses = [SimpleNamespace(
        get_velocity_dispersion=lambda value=row[0]: value,
        get_stellar_mass=lambda value=row[1]: value,
        get_einstein_radius=lambda value=row[2]: value,
        z_lens=row[3], z_source=row[4],
        get_lens_magnitude=lambda band, value=row[5]: value,
        get_source_magnitude=lambda band, value=row[6]: value,
    ) for row in data]
    background, foreground = plot_background_colors(typeplotback)
    with plot_style('white'):
        plt.rcParams.update({'text.usetex': False, 'figure.dpi': 30})
        original_settings = dict(plt.rcParams)
        figure = plotting.overview(lenses, 'F129', typeplotback=typeplotback)
        try:
            assert plotting.overplot_points(figure, lenses[:2], typeplotback=typeplotback) is None
            assert figure.get_facecolor() == to_rgba(background)
            assert len(figure.axes) == 49
            assert all(axis.xaxis.label.get_color() == foreground for axis in figure.axes)
            np.testing.assert_array_equal(plotting.lens_list_to_corner_data(lenses, 'F129'), data)
            assert dict(plt.rcParams) == original_settings
            assert any(line.get_color() == 'red' for axis in figure.axes for line in axis.lines)
            FigureCanvasAgg(figure).draw()
        finally:
            plt.close(figure)


def test_native_lens_example_theme_does_not_change_rendered_scene(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'examples' / 'native_strong_lens.py'
    specification = importlib.util.spec_from_file_location('mejiro_native_theme_example', path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    results = []
    with plot_style('white'):
        for typeplotback in ('white', 'dark'):
            filename = tmp_path / f'{typeplotback}.png'
            image = example.run_example(filename, typeplotback=typeplotback)
            assert image.backend == 'chalcedon'
            results.append(image)
            with Image.open(filename) as saved:
                expected = (0, 0, 0) if typeplotback == 'dark' else (255, 255, 255)
                assert saved.convert('RGB').getpixel((0, 0)) == expected
    for name in ('data', 'lens_surface_brightness', 'source_surface_brightness'):
        np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))