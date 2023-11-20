from astropy.io import fits


def array_to_fits(array):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=array))

    hdul.writeto('output.fits', overwrite=True)
