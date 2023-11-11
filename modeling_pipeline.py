import os
import sys
import time
from datetime import timedelta
from pprint import pprint
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy import stats

from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
import lenstronomy.Util.util as util
from lenstronomy.Util import kernel_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel

from regions import PixCoord, EllipsePixelRegion

from utils import csv_utils, psf_utils, utils

repo_path = os.getcwd()

csv_filepath = os.path.join(repo_path, 'data', 'SLACS', 'SLACS.csv')
dataset_dict_list = csv_utils.csv_to_dict_list(csv_filepath)

data_set_list = ['J9OP02010', 'J9EM25AFQ', 'J9OP04010', 'J9OP05010', 'J9EM0SEEQ', 'J9OP06010']
execution_times = []

csv_filepath = os.path.join(repo_path, 'data', 'SLACS', 'SLACS.csv')
dataset_dict_list = csv_utils.csv_to_dict_list(csv_filepath)

for data_set_name in tqdm(data_set_list):
    start_time = time.time()

    dataset = [d for d in dataset_dict_list if d.get('data_set_name') == data_set_name][0]

    target_name = dataset.get('target_name')

    figure_dir = os.path.join(repo_path, 'figures', 'pipeline', data_set_name)
    utils.create_directory_if_not_exists(figure_dir)

    ra, dec = float(dataset.get('ra')), float(dataset.get('dec'))

    with fits.open(dataset.get('cutout_filepath')) as hdu_list:
        data = hdu_list['PRIMARY'].data
        header = hdu_list['PRIMARY'].header

    wcs = WCS(header=hdu_list['PRIMARY'].header)

    center_pixel_y, center_pixel_x = wcs.all_world2pix(ra, dec, 1, adaptive=False, ra_dec_order=True)

ax = plt.subplot(projection=wcs)
ax.imshow(np.log10(data), origin='lower')
plt.grid(color='white', ls=':', alpha=0.2)
plt.scatter(center_pixel_x, center_pixel_y, edgecolor='red', facecolor='none', s=150, label='Position from MAST')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title(dataset.get('target_name'))
plt.legend()
plt.savefig(os.path.join(figure_dir, f'{data_set_name}_original_image.png'))

sky_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
size = u.Quantity((5, 5), u.arcsec)
cutout_obj = Cutout2D(data, sky_coords, size, wcs=wcs)

# overwrite data and wcs
data = cutout_obj.data
wcs = cutout_obj.wcs

print(cutout_obj.shape)

center_pixel_y, center_pixel_x = wcs.all_world2pix(ra, dec, 1, adaptive=False, ra_dec_order=True)

ax = plt.subplot(projection=wcs)
ax.imshow(np.log10(data), origin='lower')
plt.grid(color='white', ls=':', alpha=0.2)
# plt.scatter(center_pixel_x, center_pixel_y, edgecolor='red', facecolor='none', s=150, label='Position from MAST')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title(dataset.get('target_name'))
# plt.legend()
plt.savefig(os.path.join(figure_dir, f'{data_set_name}_cropped_original_image.png'))

cut_mask = stats.sigma_clip(data, sigma=2, maxiters=5)
rms = np.std(cut_mask)

print('Background RMS: ' + str(rms))

plt.imshow(cut_mask, origin='lower')
plt.title('Mask')
plt.show()
