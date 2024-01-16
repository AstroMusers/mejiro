from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


def text_boxes(ax, text_list, fontsize=18, alpha=0.5):
    props = dict(boxstyle='round', facecolor='w', alpha=alpha)
    for i, each in enumerate(ax):
        each.text(0.05, 0.95, text_list[i], transform=each.transAxes, fontsize=fontsize,
                  verticalalignment='top', bbox=props)


def source_position(ax, lens, linewidth=3):
    source_x, source_y = lens.get_source_pixel_coords()
    return ax.scatter(source_x, source_y, edgecolor='y', facecolor='none', s=150, label='Source position',
                      linewidth=linewidth)


def lens_position(ax, lens, linewidth=3):
    lens_x, lens_y = lens.get_lens_pixel_coords()
    return ax.scatter(lens_x, lens_y, edgecolor='r', facecolor='none', s=150, label='Lens position',
                      linewidth=linewidth)


def caustics(ax, lens, linewidth=3):
    _, _, ra_caustic_list, dec_caustic_list = _get_caustics_critical_curves(lens)

    x_caustic_list, y_caustic_list = [], []
    for ra, dec in zip(ra_caustic_list[0], dec_caustic_list[0]):
        x, y = lens.coords.map_coord2pix(ra=ra, dec=dec)
        x_caustic_list.append(x)
        y_caustic_list.append(y)

    return ax.plot(x_caustic_list, y_caustic_list, label='Caustics', color='g', linewidth=linewidth)


def critical_curves(ax, lens, linewidth=3):
    ra_critical_list, dec_critical_list, _, _ = _get_caustics_critical_curves(lens)

    x_critical_list, y_critical_list = [], []
    for ra, dec in zip(ra_critical_list[0], dec_critical_list[0]):
        x, y = lens.coords.map_coord2pix(ra=ra, dec=dec)
        x_critical_list.append(x)
        y_critical_list.append(y)

    return ax.plot(x_critical_list, y_critical_list, label='Critical curve', color='b', linewidth=linewidth)


def _get_caustics_critical_curves(lens):
    model_extension = LensModelExtensions(lens.lens_model_class)

    frame_size = lens.delta_pix * lens.num_pix

    return model_extension.critical_curve_caustics(lens.kwargs_lens,
                                                   compute_window=frame_size,
                                                   grid_scale=lens.delta_pix,
                                                   center_x=0.,
                                                   center_y=0.)
