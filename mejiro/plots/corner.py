import numpy as np

import corner


# def source_galaxies(lens_list, band, quantiles=[0.16, 0.5, 0.84]):
#     snr = [l.snr for l in lens_list]

#     source_R_sersic = [l.kwargs_source_dict[band]['R_sersic'] for l in lens_list]
#     # source_n_sersic = [l.kwargs_source_dict[band]['n_sersic'] for l in lens_list]
#     source_magnitude = [l.kwargs_source_dict[band]['magnitude'] for l in lens_list]
#     source_e1 = [l.kwargs_source_dict[band]['e1'] for l in lens_list]
#     source_e2 = [l.kwargs_source_dict[band]['e2'] for l in lens_list]

#     data = np.column_stack([snr, source_R_sersic, source_magnitude, source_e1, source_e2])

#     return corner.corner(
#         data,
#         labels=[
#             "SNR",
#             r"$R_\textrm{Sersic}$",
#             f'AB Mag ({band})',
#             r'$e_1$',
#             r'$e_2$',
#         ],
#         quantiles=quantiles,
#         show_titles=True
#     )


# def lens_galaxies(lens_list, band, quantiles=[0.16, 0.5, 0.84]):
#     snr = [l.snr for l in lens_list]

#     lens_R_sersic = [l.kwargs_lens_light_dict[band]['R_sersic'] for l in lens_list]
#     # lens_n_sersic = [l.kwargs_lens_light_dict[band]['n_sersic'] for l in lens_list]
#     lens_magnitude = [l.kwargs_lens_light_dict[band]['magnitude'] for l in lens_list]
#     lens_e1 = [l.kwargs_lens_light_dict[band]['e1'] for l in lens_list]
#     lens_e2 = [l.kwargs_lens_light_dict[band]['e2'] for l in lens_list]

#     data = np.column_stack([snr, lens_R_sersic, lens_magnitude, lens_e1, lens_e2])

#     return corner.corner(
#         data,
#         labels=[
#             "SNR",
#             r"$R_\textrm{Sersic}$",
#             f'AB Mag ({band})',
#             r'$e_1$',
#             r'$e_2$',
#         ],
#         quantiles=quantiles,
#         show_titles=True
#     )


def lens_list_to_corner_data(lens_list, band):
    """Convert a list of lens objects into a 2-D array suitable for corner plots.

    Extracts seven population parameters per lens: velocity dispersion, stellar
    mass, Einstein radius, lens redshift, source redshift, lens-light magnitude,
    and source magnitude.

    Parameters
    ----------
    lens_list : list
        List of lens objects, each exposing ``get_velocity_dispersion``,
        ``get_stellar_mass``, ``get_einstein_radius``, ``z_lens``,
        ``z_source``, ``get_lens_magnitude``, and ``get_source_magnitude``.
    band : str
        Photometric band used to query per-lens magnitudes (e.g. ``'F129'``).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 7)`` where *N* is the number of lenses. Columns
        are ordered as: velocity dispersion, log stellar mass, Einstein radius,
        lens redshift, source redshift, lens magnitude, source magnitude.
    """
    data = []
    for l in lens_list:
        data.append([
            l.get_velocity_dispersion(),
            l.get_stellar_mass(),
            l.get_einstein_radius(),
            l.z_lens,
            l.z_source,
            l.get_lens_magnitude(band),
            l.get_source_magnitude(band)
        ])
    return np.array(data)


def overview(lens_list, band, fig=None, quantiles=[0.16, 0.5, 0.84]):
    """Produce a corner plot overview of a lens population.

    Builds a seven-parameter corner plot (velocity dispersion, stellar mass,
    Einstein radius, lens redshift, source redshift, lens magnitude, source
    magnitude) using uniform weights so that each lens contributes equally
    regardless of sample size.

    Parameters
    ----------
    lens_list : list
        List of lens objects compatible with `lens_list_to_corner_data`.
    band : str
        Photometric band passed to `lens_list_to_corner_data` for magnitude
        extraction (e.g. ``'F129'``).
    fig : matplotlib.figure.Figure or None, optional
        Existing figure to draw into.  If ``None`` (default), a new figure is
        created by the ``corner`` library.
    quantiles : list of float, optional
        Quantiles shown as vertical lines in 1-D histograms.  Defaults to
        ``[0.16, 0.5, 0.84]`` (±1σ and median).

    Returns
    -------
    matplotlib.figure.Figure
        The corner figure object returned by ``corner.corner``.
    """
    data = lens_list_to_corner_data(lens_list, band)

    return corner.corner(
        data,
        labels = [
            r"$\sigma_v$",
            r"$\log(M_{*})$",
            r"$\theta_E$",
            r"$z_{\rm l}$",
            r"$z_{\rm s}$",
            r"$m_{\rm lens}$",
            r"$m_{\rm source}$"
        ],
        quantiles=quantiles,
        show_titles=True,
        density=True,
        weights=weights(data),
        fig=fig
    )


def overplot_points(corner_fig, lens_list):
    """Overplot individual lens positions on an existing corner figure.

    Draws a vertical dashed red line at each lens's parameter value in every
    1-D histogram panel, and plots red square markers in every 2-D scatter
    panel, allowing a small subset of lenses to be highlighted against a
    background population.

    Magnitudes are always extracted in the ``'F129'`` band.

    Parameters
    ----------
    corner_fig : matplotlib.figure.Figure
        A corner figure with a 7×7 grid of axes, as returned by `overview`.
    lens_list : list
        List of lens objects to overplot.  Must be compatible with
        `lens_list_to_corner_data`.

    Returns
    -------
    None
    """
    small_sample = lens_list_to_corner_data(lens_list, 'F129')

    axes = np.array(corner_fig.axes).reshape((7, 7))
    for i in range(7):
        for j in range(i + 1):
            for value in small_sample[:, i]:
                axes[i, j].axvline(value, color='red', linestyle='dashed', linewidth=1)
            else:
                axes[i, j].plot(small_sample[:, j], small_sample[:, i], 'rs')


def weights(data):
    """Return uniform weights that sum to one for a corner plot dataset.

    Parameters
    ----------
    data : array-like
        Dataset whose first dimension is the number of samples *N*.

    Returns
    -------
    numpy.ndarray
        1-D array of length *N* with every element equal to ``1/N``.
    """
    return 1 / len(data) * np.ones(len(data))
