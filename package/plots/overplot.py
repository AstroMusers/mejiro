from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


def source_position(ax, lens):
    source_x, source_y = lens.get_source_pixel_coords()
    
    return ax.scatter(source_x, source_y, edgecolor='y', facecolor='none', s=150, label='Source position')


def lens_position(ax, lens):
    lens_x, lens_y = lens.get_lens_pixel_coords()

    return ax.scatter(lens_x, lens_y, edgecolor='r', facecolor='none', s=150, label='Lens position')


def caustics(ax, lens):
    _, _, ra_caustic_list, dec_caustic_list = _get_caustics_critical_curves(lens)

    x_caustic_list, y_caustic_list = [], []
    for ra, dec in zip(ra_caustic_list[0], dec_caustic_list[0]):
        x, y = lens.coords.map_coord2pix(ra=ra, dec=dec)
        x_caustic_list.append(x)
        y_caustic_list.append(y)

    return ax.plot(x_caustic_list, y_caustic_list, label='Caustics', color='g')


def critical_curves(ax, lens):
    ra_critical_list, dec_critical_list, _, _ = _get_caustics_critical_curves(lens)

    x_critical_list, y_critical_list = [], []
    for ra, dec in zip(ra_critical_list[0], dec_critical_list[0]):
        x, y = lens.coords.map_coord2pix(ra=ra, dec=dec)
        x_critical_list.append(x)
        y_critical_list.append(y)

    return ax.plot(x_critical_list, y_critical_list, label='Critical curve', color='b')


def _get_caustics_critical_curves(lens):
    """_summary_

    Args:
        lens (_type_): _description_

    Returns:
        _type_: ra_critical_list
        _type_: dec_critical_list
        _type_: ra_caustic_list
        _type_: dec_caustic_list
    """
    model_extension = LensModelExtensions(lens.lens_model_class)

    frame_size = lens.delta_pix * lens.num_pix

    return model_extension.critical_curve_caustics(lens.kwargs_lens_lensing_units, 
                                                   compute_window=frame_size,
                                                   grid_scale=lens.delta_pix, 
                                                   center_x=0., 
                                                   center_y=0.)
