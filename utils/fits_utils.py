from astropy.io import fits


def get_fits_data(fits_filepath, hdu_name):
    with fits.open(fits_filepath) as hdu_list:
        hdu_list.verify()
        data = hdu_list[hdu_name].data

    return data
