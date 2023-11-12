import csv
import os
import math

import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.ImSim.image_model import ImageModel

from scipy.stats import norm, truncnorm, uniform
from tqdm import tqdm

from sim import roman, hubble


repo_path = os.getcwd()
num_images = 25
oversample_factor = 1

image_output_directory = os.path.join(repo_path, 'arrays', 'cnn_training')

# generate image filepaths
image_filepaths = [os.path.join(image_output_directory, f'{i}.png') for i in range(num_images)]

# generate random lists of parameters
# deflector model
loc = 1.2
scale = 0.2
myclip_a = 0.5
myclip_b = 2.
a = (myclip_a - loc) / scale
b = (myclip_b - loc) / scale

list_deflector_theta_E = truncnorm(a=a, b=b, loc=loc, scale=scale).rvs(size=num_images)
list_deflector_e1 = norm(loc=0.0, scale=0.1).rvs(size=num_images)
list_deflector_e2 = norm(loc=0.0, scale=0.1).rvs(size=num_images)
list_deflector_center_x = norm(loc=0.0, scale=0.16).rvs(size=num_images)
list_deflector_center_y = norm(loc=0.0, scale=0.16).rvs(size=num_images)
list_deflector_gamma1 = norm(loc=0.0, scale=0.05).rvs(size=num_images)
list_deflector_gamma2 = norm(loc=0.0, scale=0.05).rvs(size=num_images)

# deflector light
list_deflector_light_amp = uniform(loc=100, scale=10).rvs(size=num_images)
list_deflector_light_R_sersic = truncnorm(-3, 3, loc=0.5, scale=0.05).rvs(size=num_images)
list_deflector_light_n_sersic = truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(size=num_images)
list_deflector_light_e1 = norm(loc=0.0, scale=0.2).rvs(size=num_images)
list_deflector_light_e2 = norm(loc=0.0, scale=0.2).rvs(size=num_images)
list_deflector_light_center_x = list_deflector_center_x
list_deflector_light_center_y = list_deflector_center_y

# source light parameters
list_source_light_amp = uniform(loc=10, scale=1).rvs(size=num_images)
list_source_light_R_sersic = truncnorm(-2, 2, loc=0.35, scale=0.05).rvs(size=num_images)
list_source_light_n_sersic = truncnorm(-6., np.inf, loc=3., scale=0.5).rvs(size=num_images)
list_source_light_e1 = norm(loc=0.0, scale=0.1).rvs(size=num_images)
list_source_light_e2 = norm(loc=0.0, scale=0.1).rvs(size=num_images)
list_source_light_center_x = norm(loc=0.0, scale=0.16).rvs(size=num_images)
list_source_light_center_y = norm(loc=0.0, scale=0.16).rvs(size=num_images)

# classify large/small Einstein radii
list_deflector_theta_E_class = []
for einstein_radius in list_deflector_theta_E:
    if einstein_radius >= loc:
        list_deflector_theta_E_class.append('large')
    else:
        list_deflector_theta_E_class.append('small')

# lists for ra_dec at x,y=0,0
list_ra_at_xy_0, list_dec_at_xy_0 = [], []

# generate images
for i in tqdm(range(num_images)):
    # simulate main deflector
    # mass model: singular isothermal ellipsoid with a shear
    lens_model_list = ['SIE', 'SHEAR']
    kwargs_spemd = {
        'theta_E': list_deflector_theta_E[i],
        'e1': list_deflector_e1[i],
        'e2': list_deflector_e2[i],
        'center_x': list_deflector_center_x[i],
        'center_y': list_deflector_center_y[i]
    }
    kwargs_shear = {
        'gamma1': list_deflector_gamma1[i],
        'gamma2': list_deflector_gamma2[i]
    }
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list)

    # light model: sersic ellipse profile
    lens_light_model_list = ['SERSIC_ELLIPSE']
    kwargs_sersic_lens = {
        'amp': list_deflector_light_amp[i],
        'R_sersic': list_deflector_light_R_sersic[i],
        'n_sersic': list_deflector_light_n_sersic[i],
        'e1': list_deflector_light_e1[i],
        'e2': list_deflector_light_e2[i],
        'center_x': list_deflector_light_center_x[i],
        'center_y': list_deflector_light_center_y[i]
    }
    kwargs_lens_light = [kwargs_sersic_lens]
    lens_light_model_class = LightModel(lens_light_model_list)

    # simulate source
    # light model: sersic ellipse profile
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_sersic = {
        'amp': list_source_light_amp[i],
        'R_sersic': list_source_light_R_sersic[i],
        'n_sersic': list_source_light_n_sersic[i],
        'e1': list_source_light_e1[i],
        'e2': list_source_light_e2[i],
        'center_x': list_source_light_center_x[i],
        'center_y': list_source_light_center_y[i]
    }
    kwargs_source = [kwargs_sersic]
    source_model_class = LightModel(source_model_list)

    # image simulation
    kwargs_numerics = {  # TODO figure out what these parameters do
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }
    kwargs_model = {
        'lens_model_list': lens_model_list,
        'lens_light_model_list': lens_light_model_list,
        'source_light_model_list': source_model_list
    }

    side = 5  # arcseconds
    num_pix = 45 * oversample_factor
    delta_pix = side / num_pix  # size of pixel in angular coordinates

    ra_at_xy_0, dec_at_xy_0 = -delta_pix * math.ceil(num_pix / 2), -delta_pix * math.ceil(num_pix / 2) # coordinate in angles (RA/DEC) at the position of the pixel edge (0,0)
    transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix  # linear translation matrix of a shift in pixel in a shift in coordinates

    list_ra_at_xy_0.append(ra_at_xy_0)
    list_dec_at_xy_0.append(dec_at_xy_0)

    kwargs_psf = {'psf_type': 'NONE'}
    psf_class = PSF(**kwargs_psf)

    kwargs_pixel = {'nx': num_pix, 'ny': num_pix,  # number of pixels per axis
                    'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                    'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                    'transform_pix2angle': transform_pix2angle}
    pixel_grid = PixelGrid(**kwargs_pixel)

    imageModel = ImageModel(data_class=pixel_grid,
                            psf_class=psf_class,
                            lens_model_class=lens_model_class,
                            source_model_class=source_model_class,
                            lens_light_model_class=lens_light_model_class,
                            kwargs_numerics=kwargs_numerics)

    image = imageModel.image(kwargs_lens=kwargs_lens,
                            kwargs_source=kwargs_source,
                            kwargs_lens_light=kwargs_lens_light)

    # correct for spreading counts across more pixels TODO CONFIRM THIS
    # TODO does lenstronomy do this for me when I specify a different pixel grid?
    # image = image / (oversample_factor ** 2)

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
deflector_headers = ['deflector_theta_E', 'theta_E_class', 'deflector_e1', 'deflector_e2', 'deflector_center_x',
                        'deflector_center_y', 'deflector_gamma1', 'deflector_gamma2']
deflector_light_headers = ['deflector_light_R_sersic', 'deflector_light_n_sersic', 'deflector_light_e1',
                            'deflector_light_e2', 'deflector_light_center_x', 'deflector_light_center_y']
source_light_headers = ['source_light_R_sersic', 'source_light_n_sersic', 'source_light_e1', 'source_light_e2',
                        'source_light_center_x', 'source_light_center_y']
headers = ['filepath', 'ra_at_xy_0', 'dec_at_xy_0'] + deflector_headers + deflector_light_headers + source_light_headers
rows = zip(image_filepaths, list_ra_at_xy_0, list_dec_at_xy_0, list_deflector_theta_E, list_deflector_theta_E_class, list_deflector_e1,
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
