def get_pixel_psf_kwargs(kernel, supersampling_factor, kernel_point_source_normalisation=False):
    """
    Generate a dictionary for a pixel PSF that forms the keyword arguments for lenstronomy's PSF class.

    Parameters
    ----------
    kernel : array-like
        The PSF kernel array.
    supersampling_factor : int
        The supersampling factor for the point source.
    kernel_point_source_normalisation : bool, optional
        Whether to normalize the kernel for the point source, by default False.

    Returns
    -------
    dict
        A dictionary containing the PSF keyword arguments.

    Examples
    --------
    >>> from lenstronomy.Data.psf import PSF
    >>> psf_kwargs = get_pixel_psf_kwargs(kernel, supersampling_factor=5)
    >>> psf = PSF(**psf_kwargs)

    """
    return {
        'psf_type': 'PIXEL', 
        'kernel_point_source': kernel, 
        'point_source_supersampling_factor': supersampling_factor,
        'kernel_point_source_normalisation': kernel_point_source_normalisation
    }

def get_gaussian_psf_kwargs(fwhm):
    """
    Generate a dictionary for a Gaussian PSF that forms the keyword arguments for lenstronomy's PSF class.

    Parameters
    ----------
    fwhm : float or astropy.units.Quantity
        Full width at half maximum of the Gaussian PSF. If an astropy Quantity is provided, 
        its value will be extracted.

    Returns
    -------
    dict
        A dictionary containing the PSF type and the FWHM value.

    Examples
    --------
    >>> from lenstronomy.Data.psf import PSF
    >>> psf_kwargs = get_gaussian_psf_kwargs(fwhm)
    >>> psf = PSF(**psf_kwargs)
    """
    if type(fwhm) is not float:
        fwhm = fwhm.value
    return {
        'psf_type': 'GAUSSIAN', 
        'fwhm': fwhm,
    }
