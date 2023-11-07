import os

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize
from astropy.wcs import WCS

import utils


def save_plot_by_dict(dataset_dict, image_directory, filename, coords):
    cutout_filepath = dataset_dict.get('cutout_filepath')

    with fits.open(cutout_filepath) as hdu_list:
        hdu_list.verify()
        data = hdu_list['PRIMARY'].data
        header = hdu_list['PRIMARY'].header

    cutout_wcs = WCS(header)

    # calculate pixel coordinates of mast ra, dec position
    ra, dec = float(dataset_dict.get('ra')), float(dataset_dict.get('dec'))
    y_pixel, x_pixel = cutout_wcs.all_world2pix(ra, dec, 1, adaptive=False, ra_dec_order=True)

    norm = ImageNormalize(data, interval=MinMaxInterval(), stretch=SqrtStretch())
    ax = plt.subplot(projection=cutout_wcs)
    ax.imshow(data, origin='lower', norm=norm, cmap='binary')

    if coords:
        plt.grid(color='black', ls=':', alpha=0.2)
        # plt.scatter(x_pixel, y_pixel, edgecolor='red', facecolor='none', s=150, label='Position from MAST')
        plt.xlabel('Right Ascension')
        plt.ylabel('Declination')
        # plt.legend()
    else:
        plt.axis('off')

    plt.title('Target Name: ' + dataset_dict.get('target_name') + '\n' + 'Dataset: ' + dataset_dict.get('data_set_name'))

    plt.savefig(os.path.join(image_directory, filename))
    plt.close()  # don't show the image - this also dramatically speeds up execution


def plot_fits(fits_data, title):
    norm = ImageNormalize(fits_data, interval=MinMaxInterval(), stretch=SqrtStretch())

    plt.imshow(fits_data, cmap='binary', norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    plt.show()


def save_fits(fits_data, image_directory, filename, title):
    norm = ImageNormalize(fits_data, interval=MinMaxInterval(), stretch=SqrtStretch())

    plt.imshow(fits_data, cmap='binary', norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    plt.savefig(os.path.join(image_directory, filename))
