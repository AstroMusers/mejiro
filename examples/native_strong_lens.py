#!/usr/bin/env python3
"""Render an explicitly specified strong lens using Mejiro and Chalcedon only."""

import argparse
from pathlib import Path

import astropy.units as u
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from mejiro.galaxy_galaxy import GalaxyGalaxy
from mejiro.synthetic_image import SyntheticImage
from tdpy.cli import add_plot_arguments
from tdpy.plotting import save_plot, themed_plot
from tdpy.verbosity import print


class ExampleDetector:
    name = 'Illustrative detector'
    bands = ['example']

    def default_params(self):
        return {}

    def get_pixel_scale(self, band):
        return 0.05 * u.arcsec / u.pixel

    def get_zeropoint_magnitude(self, band):
        return 25.0


@themed_plot
def run_example(output_path):
    lens = GalaxyGalaxy(
        'native-example', None,
        {'lens_model_list': ['SIE', 'SHEAR'], 'lens_light_model_list': ['SERSIC_ELLIPSE'],
         'source_light_model_list': ['SERSIC_ELLIPSE'], 'lens_redshift_list': [0.5, 0.5],
         'source_redshift_list': [1.5], 'z_source': 1.5, 'cosmo': Planck18},
        {'kwargs_lens': [{'theta_E': 1.1, 'e1': 0.2, 'e2': 0.05}, {'gamma1': 0.03, 'gamma2': 0.02}],
         'kwargs_lens_light': [{'R_sersic': 0.4, 'n_sersic': 3.0, 'e1': 0.2, 'e2': 0.05,
                                'center_x': 0.0, 'center_y': 0.0}],
         'kwargs_source': [{'R_sersic': 0.12, 'n_sersic': 1.0, 'e1': 0.1, 'e2': -0.08,
                            'center_x': 0.15, 'center_y': 0.04}], 'kwargs_ps': []},
        physical_params={'magnitudes': {'lens': {'example': 21.0}, 'source': {'example': 24.0}}},
        backend='chalcedon',
    )
    image = SyntheticImage(lens, ExampleDetector(), 'example', fov_arcsec=4.0, pieces=True,
                           kwargs_psf={'psf_type': 'GAUSSIAN', 'fwhm': 0.12})
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=True, sharey=True,
                                constrained_layout=True, facecolor='white')
    extent = [-0.5 * image.fov_arcsec, 0.5 * image.fov_arcsec] * 2
    for axis, values, title in zip(axes,
                                   (image.lens_surface_brightness, image.source_surface_brightness, image.data),
                                   ('Lens galaxy', 'Lensed source', 'Combined scene')):
        maximum = values.max()
        rendered = axis.imshow(np.maximum(values, maximum * 1e-5), origin='lower', extent=extent,
                               cmap='gray_r', norm=LogNorm(maximum * 1e-5, maximum))
        axis.set(title=title, xlabel='Angle [arcsec]')
        axis.grid(False)
        figure.colorbar(rendered, ax=axis, label='Count rate [s$^{-1}$ pixel$^{-1}$]', shrink=0.8)
    axes[0].set_ylabel('Angle [arcsec]')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing to {output_path}...')
    save_plot(figure, output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return image


def main():
    parser = add_plot_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    arguments = parser.parse_args()
    run_example(Path(__file__).with_name(f'native_strong_lens.{arguments.typefileplot}'), typeplotback=arguments.typeplotback)


if __name__ == '__main__':
    main()