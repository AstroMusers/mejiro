import os
import warnings
from datetime import datetime

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from tqdm import tqdm

from package.utils import util


def produce_cutouts(collection_name, data_path, dataset_list, num_pix):

    print('Cropping to ' + str(num_pix) + ' pixels')

    # create a directory in cutouts directory if one doesn't already exist
    cutout_directory = os.path.join(data_path, 'cutouts', collection_name)
    utils.create_directory_if_not_exists(cutout_directory)

    for dataset in tqdm(dataset_list):
        dataset_name = dataset.get('data_set_name')
        mast_ra = float(dataset.get('ra'))
        mast_dec = float(dataset.get('dec'))
        fits_filepath = dataset.get('fits_filepath')

        if fits_filepath is None:
            message = 'Could not identify file for dataset ' + dataset_name
            warnings.warn(message)
            continue
        else:
            with fits.open(fits_filepath) as hdu_list:
                hdu_list.verify()
                data = hdu_list['SCI'].data
                primary_header = hdu_list['PRIMARY'].header
                header = hdu_list['SCI'].header

            # TODO get to the bottom of the error I'm circumventing: "ValueError: an astropy.io.fits.HDUList is required for Lookup table distortion."
            try:
                wcs = WCS(header=hdu_list['SCI'].header)

                mast_coords = SkyCoord(mast_ra, mast_dec, unit='deg', frame='icrs')
                size = u.Quantity((num_pix, num_pix), u.pixel)
                cutout_obj = Cutout2D(data, mast_coords, size, wcs=wcs)  # NB this has updated wcs

                cutout_filename = 'cutout_' + dataset.get('aper') + '_' + dataset.get('spec') + '_' + dataset_name + '.fits'
                cutout_filepath = os.path.join(cutout_directory, cutout_filename)

                # update wcs in header
                cutout_header = header
                cutout_header.update(cutout_obj.wcs.to_header())
                cutout_header['COMMENT'] = ' = \'Cropped fits file ' + str(datetime.now()) + '\''

                # fits files contain the D001SCAL keyword which gives the pixel scale in arcseconds
                # this is stored in primary HDU but not SCI, so get from primary and add to cutout_header
                cutout_header['D001SCAL'] = primary_header['D001SCAL']

                # save cutout fits file
                hdu = fits.PrimaryHDU(cutout_obj.data, header=cutout_header)
                hdu.writeto(cutout_filepath, overwrite=True)

                # update the dataset with the filepath to the cutout
                dataset['cutout_filepath'] = cutout_filepath

            except ValueError as e:
                warnings.warn(str(e))
                continue

    # return updated dict_list
    return dataset_list
