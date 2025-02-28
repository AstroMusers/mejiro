def get_pixel_psf_kwargs(kernel, supersampling_factor, kernel_point_source_normalisation=False):
    return {
        'psf_type': 'PIXEL', 
        'kernel_point_source': kernel, 
        'point_source_supersampling_factor': supersampling_factor,
        'kernel_point_source_normalisation': kernel_point_source_normalisation
    }

def get_gaussian_psf_kwargs(fwhm):
    return {
        'psf_type': 'GAUSSIAN', 
        'fwhm': fwhm,
    }
