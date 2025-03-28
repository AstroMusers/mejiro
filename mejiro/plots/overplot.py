from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


# TODO all of these should be moved to the plotting module and be options for plotting any lens

def text_boxes(ax, text_list, alpha=0.5):
    props = dict(boxstyle='round', facecolor='w', alpha=alpha)
    for i, each in enumerate(ax):
        each.text(0.05, 0.95, text_list[i], transform=each.transAxes,
                  verticalalignment='top', bbox=props)


def source_position(ax, lens, coords, alpha=1, color='y', size=100):
    source_x, source_y = lens.get_source_pixel_coords(coords)
    return ax.plot(source_x, source_y, marker='o', color=color, markersize=size, label='Source', alpha=alpha)


def lens_position(ax, lens, coords, alpha=1, color='r', size=100):
    lens_x, lens_y = lens.get_lens_pixel_coords(coords)
    return ax.plot(lens_x, lens_y, marker='o', color=color, markersize=size, label='Lens', alpha=alpha)


def caustics(ax, lens, coords, num_pix, delta_pix=0.11, linewidth=2, alpha=1, color='g'):
    _, _, ra_caustic_list, dec_caustic_list = _get_caustics_critical_curves(lens, num_pix, delta_pix)

    x_caustic_list, y_caustic_list = [], []
    for ra, dec in zip(ra_caustic_list[0], dec_caustic_list[0]):
        x, y = coords.map_coord2pix(ra=ra, dec=dec)
        x_caustic_list.append(x)
        y_caustic_list.append(y)

    return ax.plot(x_caustic_list, y_caustic_list, label='Caustic', color=color, linewidth=linewidth, alpha=alpha)


def critical_curves(ax, lens, coords, num_pix, delta_pix=0.11, linewidth=2, alpha=1, color='b'):
    ra_critical_list, dec_critical_list, _, _ = _get_caustics_critical_curves(lens, num_pix, delta_pix)

    x_critical_list, y_critical_list = [], []
    for ra, dec in zip(ra_critical_list[0], dec_critical_list[0]):
        x, y = coords.map_coord2pix(ra=ra, dec=dec)
        x_critical_list.append(x)
        y_critical_list.append(y)

    return ax.plot(x_critical_list, y_critical_list, label='Critical', color=color, linewidth=linewidth,
                   alpha=alpha)


# TODO this is all a bit messy
def _get_caustics_critical_curves(lens, num_pix, delta_pix):
    model_extension = LensModelExtensions(lens.lens_model_class)

    frame_size = delta_pix * num_pix
    return model_extension.critical_curve_caustics(lens.kwargs_lens,
                                                   compute_window=frame_size,
                                                   grid_scale=delta_pix,
                                                   center_x=0.,
                                                   center_y=0.)
