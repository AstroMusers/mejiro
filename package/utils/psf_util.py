from lenstronomy.Data.psf import PSF


def get_pixel_kwargs_psf(kernel):
    return {
        'psf_type': 'PIXEL',
        'kernel_point_source': kernel
    }


def get_none_kwargs_psf():
    return {
        'psf_type': 'NONE'
    }


def get_gaussian_kwargs_psf(fwhm, pixel_size, truncation):
    return {
        'psf_type': 'GAUSSIAN',
        'fwhm': fwhm,  # float, full width at half maximum
        'pixel_size': pixel_size,  # width of pixel
        'truncation': truncation  # float, Gaussian truncation (in units of sigma)
    }


def get_psf_class(kwargs_psf):
    return PSF(**kwargs_psf)
