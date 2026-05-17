from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


# TODO all of these should be moved to the plotting module and be options for plotting any lens

def text_boxes(ax, text_list, alpha=0.5):
    """Annotate each axis in a list with a rounded text box in the upper-left.

    Parameters
    ----------
    ax : array-like of matplotlib.axes.Axes
        Axes to annotate, one per entry in *text_list*.
    text_list : list of str
        Annotation strings, one per axis.
    alpha : float, optional
        Background transparency of the text box.  Defaults to ``0.5``.

    Returns
    -------
    None
    """
    props = dict(boxstyle='round', facecolor='w', alpha=alpha)
    for i, each in enumerate(ax):
        each.text(0.05, 0.95, text_list[i], transform=each.transAxes,
                  verticalalignment='top', bbox=props)


def source_position(ax, lens, coords, alpha=1, color='y', size=100):
    """Overplot the source position as a circular marker on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the marker.
    lens : StrongLens
        Lens object whose ``get_source_pixel_coords`` method returns the pixel
        coordinates of the source centre.
    coords : lenstronomy.ImSim.Numerics.grid.RegularGrid or similar
        Coordinate mapping object passed to ``get_source_pixel_coords``.
    alpha : float, optional
        Marker opacity.  Defaults to ``1``.
    color : str, optional
        Marker color.  Defaults to ``'y'`` (yellow).
    size : float, optional
        Marker size in points.  Defaults to ``100``.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Plot handle returned by ``ax.plot``.
    """
    source_x, source_y = lens.get_source_pixel_coords(coords)
    return ax.plot(source_x, source_y, marker='o', color=color, markersize=size, label='Source', alpha=alpha)


def lens_position(ax, lens, coords, alpha=1, color='r', size=100):
    """Overplot the lens position as a circular marker on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the marker.
    lens : StrongLens
        Lens object whose ``get_lens_pixel_coords`` method returns the pixel
        coordinates of the lens centre.
    coords : lenstronomy.ImSim.Numerics.grid.RegularGrid or similar
        Coordinate mapping object passed to ``get_lens_pixel_coords``.
    alpha : float, optional
        Marker opacity.  Defaults to ``1``.
    color : str, optional
        Marker color.  Defaults to ``'r'`` (red).
    size : float, optional
        Marker size in points.  Defaults to ``100``.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Plot handle returned by ``ax.plot``.
    """
    lens_x, lens_y = lens.get_lens_pixel_coords(coords)
    return ax.plot(lens_x, lens_y, marker='o', color=color, markersize=size, label='Lens', alpha=alpha)


def caustics(ax, lens, coords, num_pix, delta_pix=0.11, linewidth=2, alpha=1, color='g'):
    """Overplot the caustic curve on an axis in pixel coordinates.

    Computes the caustic via ``_get_caustics_critical_curves``, converts the
    RA/Dec positions to pixels using *coords*, and draws the result as a line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the caustic.
    lens : StrongLens
        Lens object passed to ``_get_caustics_critical_curves``.
    coords : lenstronomy.ImSim.Numerics.grid.RegularGrid or similar
        Coordinate mapping object used to convert RA/Dec to pixel positions.
    num_pix : int
        Number of pixels along one side of the image, used to set the
        computation window size.
    delta_pix : float, optional
        Pixel scale in arcseconds.  Defaults to ``0.11``.
    linewidth : float, optional
        Width of the plotted line.  Defaults to ``2``.
    alpha : float, optional
        Line opacity.  Defaults to ``1``.
    color : str, optional
        Line color.  Defaults to ``'g'`` (green).

    Returns
    -------
    list of matplotlib.lines.Line2D
        Plot handle returned by ``ax.plot``.
    """
    _, _, ra_caustic_list, dec_caustic_list = _get_caustics_critical_curves(lens, num_pix, delta_pix)

    x_caustic_list, y_caustic_list = [], []
    for ra, dec in zip(ra_caustic_list[0], dec_caustic_list[0]):
        x, y = coords.map_coord2pix(ra=ra, dec=dec)
        x_caustic_list.append(x)
        y_caustic_list.append(y)

    return ax.plot(x_caustic_list, y_caustic_list, label='Caustic', color=color, linewidth=linewidth, alpha=alpha)


def critical_curves(ax, lens, coords, num_pix, delta_pix=0.11, linewidth=2, alpha=1, color='b'):
    """Overplot the critical curve on an axis in pixel coordinates.

    Computes the critical curve via ``_get_caustics_critical_curves``, converts
    the RA/Dec positions to pixels using *coords*, and draws the result as a
    line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the critical curve.
    lens : StrongLens
        Lens object passed to ``_get_caustics_critical_curves``.
    coords : lenstronomy.ImSim.Numerics.grid.RegularGrid or similar
        Coordinate mapping object used to convert RA/Dec to pixel positions.
    num_pix : int
        Number of pixels along one side of the image, used to set the
        computation window size.
    delta_pix : float, optional
        Pixel scale in arcseconds.  Defaults to ``0.11``.
    linewidth : float, optional
        Width of the plotted line.  Defaults to ``2``.
    alpha : float, optional
        Line opacity.  Defaults to ``1``.
    color : str, optional
        Line color.  Defaults to ``'b'`` (blue).

    Returns
    -------
    list of matplotlib.lines.Line2D
        Plot handle returned by ``ax.plot``.
    """
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
    """Compute critical curves and caustics for a lens using lenstronomy.

    Parameters
    ----------
    lens : StrongLens
        Lens object providing ``lens_model_class`` and ``kwargs_lens``.
    num_pix : int
        Number of pixels along one side of the image; multiplied by
        *delta_pix* to obtain the computation window size in arcseconds.
    delta_pix : float
        Pixel scale in arcseconds, used as the grid scale for the computation.

    Returns
    -------
    ra_critical_list : list of numpy.ndarray
        RA coordinates of critical curve points.
    dec_critical_list : list of numpy.ndarray
        Dec coordinates of critical curve points.
    ra_caustic_list : list of numpy.ndarray
        RA coordinates of caustic points.
    dec_caustic_list : list of numpy.ndarray
        Dec coordinates of caustic points.
    """
    model_extension = LensModelExtensions(lens.lens_model_class)

    frame_size = delta_pix * num_pix
    return model_extension.critical_curve_caustics(lens.kwargs_lens,
                                                   compute_window=frame_size,
                                                   grid_scale=delta_pix,
                                                   center_x=0.,
                                                   center_y=0.)
