import csv
import os
import math

import matplotlib.pyplot as plt
import numpy as np

from lenstronomy.Data.psf import PSF
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.ImSim.image_model import ImageModel

from tqdm import tqdm

from params import generate

repo_path = os.getcwd()  # TODO fix
num_images = 25
oversample_factor = 1

image_output_directory = os.path.join(repo_path, 'output', 'arrays', 'cnn_training')

# generate image filepaths
image_filepaths = [os.path.join(image_output_directory, f'model_{i}.png') for i in range(num_images)]

# lists for ra_dec at x,y=0,0
list_ra_at_xy_0, list_dec_at_xy_0 = [], []

# get params class
model = generate.simple.Simple(num=num_images)

# image simulation
kwargs_numerics = {
    'supersampling_factor': 1,
    'supersampling_convolution': False
}

side = 5  # arcseconds
num_pix = 45 * oversample_factor
delta_pix = side / num_pix  # size of pixel in angular coordinates

ra_at_xy_0, dec_at_xy_0 = -delta_pix * math.ceil(num_pix / 2), -delta_pix * math.ceil(
    num_pix / 2)  # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
transform_pix2angle = np.array(
    [[1, 0], [0, 1]]) * delta_pix  # linear translation matrix of a shift in pixel in a shift in coordinates

list_ra_at_xy_0.append(ra_at_xy_0)
list_dec_at_xy_0.append(dec_at_xy_0)

kwargs_psf = {'psf_type': 'NONE'}
psf_class = PSF(**kwargs_psf)

kwargs_pixel = {
    'nx': num_pix,
    'ny': num_pix,  # number of pixels per axis
    'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
    'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
    'transform_pix2angle': transform_pix2angle
}
pixel_grid = PixelGrid(**kwargs_pixel)

# generate images
for i in tqdm(range(num_images)):

    imageModel = ImageModel(data_class=pixel_grid,
                            psf_class=psf_class,
                            lens_model_class=model.lens_model_class,
                            source_model_class=model.source_model_class,
                            lens_light_model_class=model.lens_light_model_class,
                            kwargs_numerics=kwargs_numerics)

    image = imageModel.image(kwargs_lens=model.get_kwargs_lens(i),
                             kwargs_source=model.get_kwargs_source(i),
                             kwargs_lens_light=model.get_kwargs_lens_light(i))

    # save array
    array_path = os.path.join(image_output_directory, f'{i}.npy')
    np.save(array_path, image)

    # save image
    image_name = os.path.basename(image_filepaths[i])
    # plt.imshow(roman_image, aspect='equal', origin='lower')
    plt.imshow(image, aspect='equal', origin='lower')
    plt.axis('off')
    plt.savefig(os.path.join(image_output_directory, image_name), bbox_inches='tight', pad_inches=0)
    plt.close()

# write all parameters to CSV



headers = ['filepath', 'ra_at_xy_0', 'dec_at_xy_0'] + deflector_headers + deflector_light_headers + source_light_headers
rows = zip(image_filepaths, list_ra_at_xy_0, list_dec_at_xy_0, list_deflector_theta_E, list_deflector_theta_E_class,
           list_deflector_e1,
           list_deflector_e2, list_deflector_center_x, list_deflector_center_y, list_deflector_gamma1,
           list_deflector_gamma2, list_deflector_light_R_sersic, list_deflector_light_n_sersic,
           list_deflector_light_e1, list_deflector_light_e2, list_deflector_light_center_x,
           list_deflector_light_center_y, list_source_light_R_sersic, list_source_light_n_sersic,
           list_source_light_e1, list_source_light_e2, list_source_light_center_x, list_source_light_center_y)

with open(os.path.join(image_output_directory, 'truths.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
