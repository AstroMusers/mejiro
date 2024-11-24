import corner
import numpy as np


def source_galaxies(lens_list, band, quantiles=[0.16, 0.5, 0.84]):
    snr = [l.snr for l in lens_list]

    source_R_sersic = [l.kwargs_source_dict[band]['R_sersic'] for l in lens_list]
    # source_n_sersic = [l.kwargs_source_dict[band]['n_sersic'] for l in lens_list]
    source_magnitude = [l.kwargs_source_dict[band]['magnitude'] for l in lens_list]
    source_e1 = [l.kwargs_source_dict[band]['e1'] for l in lens_list]
    source_e2 = [l.kwargs_source_dict[band]['e2'] for l in lens_list]

    data = np.column_stack([snr, source_R_sersic, source_magnitude, source_e1, source_e2])

    return corner.corner(
        data,
        labels=[
            "SNR",
            r"$R_\textrm{Sersic}$",
            f'AB Mag ({band})',
            r'$e_1$',
            r'$e_2$',
        ],
        quantiles=quantiles,
        show_titles=True
    )


def lens_galaxies(lens_list, band, quantiles=[0.16, 0.5, 0.84]):
    snr = [l.snr for l in lens_list]

    lens_R_sersic = [l.kwargs_lens_light_dict[band]['R_sersic'] for l in lens_list]
    # lens_n_sersic = [l.kwargs_lens_light_dict[band]['n_sersic'] for l in lens_list]
    lens_magnitude = [l.kwargs_lens_light_dict[band]['magnitude'] for l in lens_list]
    lens_e1 = [l.kwargs_lens_light_dict[band]['e1'] for l in lens_list]
    lens_e2 = [l.kwargs_lens_light_dict[band]['e2'] for l in lens_list]

    data = np.column_stack([snr, lens_R_sersic, lens_magnitude, lens_e1, lens_e2])

    return corner.corner(
        data,
        labels=[
            "SNR",
            r"$R_\textrm{Sersic}$",
            f'AB Mag ({band})',
            r'$e_1$',
            r'$e_2$',
        ],
        quantiles=quantiles,
        show_titles=True
    )


def system(lens_list, band, quantiles=[0.16, 0.5, 0.84]):
    snr = [l.snr for l in lens_list]
    z_lens = [l.z_lens for l in lens_list]
    z_source = [l.z_source for l in lens_list]
    lens_mag = [l.lens_mags[band] for l in lens_list]
    source_mag = [l.source_mags[band] for l in lens_list]
    einstein_radius = [l.get_einstein_radius() for l in lens_list]
    # velocity_dispersion = [l.lens_vel_disp for l in lens_list]

    data = np.column_stack([snr, z_lens, z_source, lens_mag, source_mag, einstein_radius])  # velocity_dispersion

    return corner.corner(
        data,
        labels=[
            "SNR",
            r"$z_\textrm{lens}$",
            r"$z_\textrm{source}$",
            r"$m_\textrm{lens}$" + f" ({band})",
            r"$m_\textrm{source}$" + f" ({band})",
            r'$\theta_\textrm{E}$',
            # r'$\sigma_\textrm{v}$',
        ],
        quantiles=quantiles,
        show_titles=True,
        density=True
    )
