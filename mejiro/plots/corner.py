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
    small_sample = lens_list_to_corner_data(lens_list, 'F129')

    axes = np.array(corner_fig.axes).reshape((7, 7))
    for i in range(7):
        for j in range(i + 1):
            for value in small_sample[:, i]:
                axes[i, j].axvline(value, color='red', linestyle='dashed', linewidth=1)
            else:
                axes[i, j].plot(small_sample[:, j], small_sample[:, i], 'rs')


def weights(data):
    return 1 / len(data) * np.ones(len(data))
