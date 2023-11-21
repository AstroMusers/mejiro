import os
import pickle
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'

from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy import stats
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from regions.shapes.circle import CirclePixelRegion
from photutils.detection import DAOStarFinder
from regions import CircleAnnulusPixelRegion
from regions import PixCoord

from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Util import kernel_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.pixel_grid import PixelGrid

from package.utils import csv_util, psf_util, util

repo_path = os.getcwd()

csv_filepath = os.path.join(repo_path, 'data', 'SLACS', 'SLACS.csv')
dataset_dict_list = csv_utils.csv_to_dict_list(csv_filepath)

data_set_list = ['J9EM0SEEQ']  # 'J9OP02010', 'J9EM25AFQ', 'J9OP04010', 'J9OP05010', 'J9EM0SEEQ', 'J9OP06010'
execution_times = []

csv_filepath = os.path.join(repo_path, 'data', 'SLACS', 'SLACS.csv')
dataset_dict_list = csv_utils.csv_to_dict_list(csv_filepath)

for data_set_name in tqdm(data_set_list):

    execution_start_time = time.time()

    dataset = [d for d in dataset_dict_list if d.get('data_set_name') == data_set_name][0]

    target_name = dataset.get('target_name')

    figure_dir = os.path.join(repo_path, 'figures', 'model', data_set_name)
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
    plt.close()

    sky_coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    size = u.Quantity((5, 5), u.arcsec)
    cutout_obj = Cutout2D(data, sky_coords, size, wcs=wcs)

    # overwrite data and wcs
    data = cutout_obj.data
    wcs = cutout_obj.wcs

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
    plt.close()

    cut_mask = stats.sigma_clip(data, sigma=2, maxiters=5)
    rms = np.std(cut_mask)

    plt.imshow(cut_mask, origin='lower')
    plt.title('Calculating background RMS')
    plt.savefig(os.path.join(figure_dir, f'{data_set_name}_rms_calculation.png'))
    plt.close()

    # masking
    if data_set_name not in ['J9OP05010']:

        mean, median, std = sigma_clipped_stats(data, sigma=3.)

        daofind = DAOStarFinder(fwhm=1.2, threshold=5 * std)
        sources = daofind(data - median)

        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=4.0)
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
                   interpolation='nearest')
        apertures.plot(color='blue', lw=1.5, alpha=0.5)
        plt.title(dataset.get('target_name'))
        plt.savefig(os.path.join(figure_dir, f'{data_set_name}_points_to_mask.png'))
        plt.close()

        # CUSTOM MASK FOR WUPRS LENS
        mask_center_x = 28
        mask_center_y = 15

        # build mask
        center = PixCoord(mask_center_x, mask_center_y)
        radius = 2.5

        # get mean of region around
        annulus = CircleAnnulusPixelRegion(center=center, inner_radius=radius, outer_radius=4)
        annulus_array = annulus.to_mask(mode='center').to_image(data.shape)
        annulus_array *= data
        annulus_pixels = annulus_array[annulus_array != 0]
        annulus_mean = np.mean(annulus_pixels)

        region = CirclePixelRegion(center=center, radius=radius)
        mask = region.to_mask(mode='center')
        mask_array = mask.to_image(data.shape)  # returns 2d np array with ones for masked pixels, else zero
        for row_num, row in enumerate(mask_array):
            for item_num, item in enumerate(row):
                if item == 1:
                    mask_array[row_num][item_num] = 0
                if item == 0:
                    mask_array[row_num][item_num] = 1
        data = data * mask_array  # set masked pixels to zero
        array_to_add = mask.to_image(data.shape)
        array_to_add[array_to_add == 1] = annulus_mean
        data += array_to_add

        # identify those bright spots well outside of the centered lensing galaxy
        num_sources_masked = 0
        masks = []
        for source in sources:
            if source['xcentroid'] > 60 or source['xcentroid'] < 40 or source['ycentroid'] > 60 or source[
                'ycentroid'] < 40:
                mask_center_x = source['xcentroid']
                mask_center_y = source['ycentroid']

                # build mask
                center = PixCoord(mask_center_x, mask_center_y)
                radius = 2

                # get mean of region around
                annulus = CircleAnnulusPixelRegion(center=center, inner_radius=radius, outer_radius=4)
                annulus_array = annulus.to_mask(mode='center').to_image(data.shape)
                annulus_array *= data
                annulus_pixels = annulus_array[annulus_array != 0]
                annulus_mean = np.mean(annulus_pixels)

                region = CirclePixelRegion(center=center, radius=radius)
                mask = region.to_mask(mode='center')
                mask_array = mask.to_image(data.shape)  # returns 2d np array with ones for masked pixels, else zero
                for row_num, row in enumerate(mask_array):
                    for item_num, item in enumerate(row):
                        if item == 1:
                            mask_array[row_num][item_num] = 0
                        if item == 0:
                            mask_array[row_num][item_num] = 1
                data = data * mask_array  # set masked pixels to zero
                array_to_add = mask.to_image(data.shape)
                array_to_add[array_to_add == 1] = annulus_mean
                data += array_to_add

                num_sources_masked += 1

        plt.imshow(data)
        plt.title(dataset.get('target_name'))
        plt.colorbar()
        plt.savefig(os.path.join(figure_dir, f'{data_set_name}_masked.png'))
        plt.close()

    # data specifics
    start_time = datetime.fromisoformat(dataset.get('start_time'))
    stop_time = datetime.fromisoformat(dataset.get('stop_time'))
    background_rms = rms  # background noise per pixel
    exp_time = (
            stop_time - start_time).seconds  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    pixel_scale = float(header.get('D001SCAL'))  # pixel size in arcsec (area per pixel = pixel_scale**2)

    # read out matrix elements and convert them in units of arc seconds
    CD1_1 = header.get('CD1_1') * 3600  # change in arc sec per pixel d(ra)/dx
    CD1_2 = header.get('CD1_2') * 3600
    CD2_1 = header.get('CD2_1') * 3600
    CD2_2 = header.get('CD2_2') * 3600

    # generate pixel-to-coordinate transform matrix and its inverse
    pix2coord_transform_undistorted = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
    det = CD1_1 * CD2_2 - CD1_2 * CD2_1
    coord2pix_transform_undistorted = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]]) / det

    # read out pixel size of image
    nx, ny = data.shape
    x_c = int(nx / 2)
    y_c = int(ny / 2)

    # compute RA/DEC relative shift between the edge and the center of the image
    dra, ddec = pix2coord_transform_undistorted.dot(np.array([x_c, y_c]))

    # set edge of the image such that the center has RA/DEC = (0,0)
    ra_at_xy_0, dec_at_xy_0 = -dra, -ddec

    psf_filepath = os.path.join(repo_path, 'psfs', 'PSFSTD_WFC3UV_F814W.fits')

    with fits.open(psf_filepath, ignore_missing_end=True) as hdu_list:
        # hdu_list.verify()
        psf_data = hdu_list['PRIMARY'].data[17]  # middle of first chip
        psf_header = hdu_list['PRIMARY'].header

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax1.matshow(np.log10(psf_data), origin='lower')
    ax2.matshow(psf_data, origin='lower')
    title1 = 'Log10'
    ax1.set_title(title1)
    title2 = 'Raw'
    ax2.set_title(title2)
    plt.savefig(os.path.join(figure_dir, f'{data_set_name}_psf'))
    plt.close()

    # if kernel needs to be cut down, lenstronomy has a method for that
    kernel_size = psf_header.get('NAXIS1')  # PSF kernel size (odd number required).
    kernel_cut = kernel_util.cut_psf(psf_data, kernel_size)
    psf_pix_map = kernel_util.degrade_kernel(psf_data - np.min(psf_data), 2)

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_pix_map,
        'point_source_supersampling_factor': 2
    }
    psf_class = psf_utils.get_psf_class(kwargs_psf)

    kwargs_data = {'background_rms': background_rms,  # rms of background noise
                   'exposure_time': exp_time,  # exposure time (or a map per pixel)
                   'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
                   'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel
                   'transform_pix2angle': pix2coord_transform_undistorted,
                   # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix), units of arcseconds
                   'image_data': data
                   }

    data_class = ImageData(**kwargs_data)
    data_class.update_data(data)

    # TODO insert
    from params import modeling

    model_params = modeling.simple

    kwargs_params = {'lens_model': model_params.lens_params,
                     'source_model': model_params.source_params,
                     'lens_light_model': model_params.lens_light_params}  # NB add special params here if using them

    kwargs_model = {'lens_model_list': model_params.lens_model_list,
                    'source_light_model_list': model_params.source_model_list,
                    'lens_light_model_list': model_params.lens_light_model_list}

    kwargs_likelihood = {'source_marg': False}
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
    # if you have multiple bands to be modeled simultaneously, you can append them to the multi_band_list

    kwargs_data_joint = {'multi_band_list': multi_band_list,
                         'multi_band_type': 'single-band'
                         # 'multi-linear': every imaging band has independent solutions of the surface brightness, 'joint-linear': there is one joint solution of the linear coefficients demanded across the bands.
                         }
    kwargs_constraints = {
        'linear_solver': True}  # optional, if 'linear_solver': False, lenstronomy does not apply a linear inversion of the 'amp' parameters during fitting but instead samples them.

    kwargs_pixel = {'nx': nx, 'ny': ny,  # number of pixels per axis
                    'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                    'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                    'transform_pix2angle': pix2coord_transform_undistorted}
    pixel_grid = PixelGrid(**kwargs_pixel)

    lens_model_class = LensModel(lens_model_list=model_params.lens_model_list)
    source_model_class = LightModel(light_model_list=model_params.source_model_list)
    lens_light_model_class = LightModel(light_model_list=model_params.lens_light_model_list)

    imageModel = ImageModel(data_class=pixel_grid,
                            psf_class=psf_class,
                            lens_model_class=lens_model_class,
                            source_model_class=source_model_class,
                            lens_light_model_class=lens_light_model_class,
                            kwargs_numerics=kwargs_numerics)

    # generate image
    image_sim = imageModel.image(kwargs_lens=model_params.kwargs_lens_init,
                                 kwargs_source=model_params.kwargs_source_init,
                                 kwargs_lens_light=model_params.kwargs_lens_light_init)

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax1.matshow(image_sim, origin='lower')
    ax2.imshow(data, origin='lower')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    title1 = 'Initial Parameters'
    ax1.set_title(title1)
    title2 = 'Image'
    ax2.set_title(title2)
    plt.savefig(os.path.join(figure_dir, f'{data_set_name}_fitting_params.png'))
    plt.close()

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

    # pso = ['PSO', {'sigma_scale': 1., 'n_particles': 100, 'n_iterations': 100}]
    pso = ['PSO', {'sigma_scale': 1., 'n_particles': 200, 'n_iterations': 200}]
    # pso = ['PSO', {'sigma_scale': 1., 'n_particles': 400, 'n_iterations': 400}]
    # mcmc = ['MCMC', {'n_burn': 20, 'n_run': 20, 'walkerRatio': 4, 'sigma_scale': .1}]
    mcmc = ['MCMC', {'n_burn': 100, 'n_run': 100, 'walkerRatio': 10, 'sigma_scale': .1}]
    # mcmc = ['MCMC', {'n_burn': 200, 'n_run': 600, 'n_walkers': 200, 'sigma_scale': .1}]
    fitting_kwargs_list = [pso, mcmc]

    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()

    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                          linear_solver=kwargs_constraints.get('linear_solver', True))

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex='none', sharey='none')

    modelPlot.data_plot(ax=axes[0, 0])
    modelPlot.model_plot(ax=axes[0, 1])
    modelPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
    modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=pixel_scale, numPix=nx)
    modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
    modelPlot.magnification_plot(ax=axes[1, 2])
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    # f.show()
    f.savefig(os.path.join(figure_dir, f'{data_set_name}_p1.png'))
    plt.close()

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex='none', sharey='none')

    modelPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)
    modelPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)
    modelPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)
    modelPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)
    modelPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                                 unconvolved=True)
    modelPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True, lens_light_add=True,
                                 point_source_add=True)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    # f.show()
    f.savefig(os.path.join(figure_dir, f'{data_set_name}_p2.png'))
    plt.close()

    sampler_type, samples_mcmc, param_mcmc, dist_mcmc = chain_list[1]
    param_class = fitting_seq.param_class

    i = 0
    for i in range(len(chain_list)):
        f, axes = chain_plot.plot_chain_list(chain_list, i)
        f.savefig(os.path.join(figure_dir, f'{data_set_name}_p' + str(i + 3) + '.png'))
        plt.close()

    n_sample = len(samples_mcmc)
    samples_mcmc_cut = samples_mcmc[int(n_sample * 1 / 2.):]

    kwargs_macromodel_lens = kwargs_result.get('kwargs_lens')
    kwargs_macromodel_lens_light = kwargs_result.get('kwargs_lens_light')
    kwargs_macromodel_ps = kwargs_result.get('kwargs_ps')
    kwargs_macromodel_source = kwargs_result.get('kwargs_source')
    kwargs_macromodel_special = kwargs_result.get('kwargs_special')

    cache_path = os.path.join(repo_path, 'data', 'cache')

    modeled_lenses_dir = os.path.join(repo_path, 'data', 'modeled_lenses', data_set_name)
    utils.create_directory_if_not_exists(modeled_lenses_dir)

    with open(os.path.join(modeled_lenses_dir, data_set_name + '_lens'), 'ab') as lens_file:
        pickle.dump(kwargs_macromodel_lens, lens_file)

    with open(os.path.join(modeled_lenses_dir, data_set_name + '_lens_light'), 'ab') as lens_light_file:
        pickle.dump(kwargs_macromodel_lens_light, lens_light_file)

    with open(os.path.join(modeled_lenses_dir, data_set_name + '_source'), 'ab') as source_file:
        pickle.dump(kwargs_macromodel_source, source_file)

    lens_model_class = LensModel(lens_model_list=model_params.lens_model_list)
    source_model_class = LightModel(light_model_list=model_params.source_model_list)
    lens_light_model_class = LightModel(light_model_list=model_params.lens_light_model_list)

    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                            kwargs_numerics=kwargs_numerics)

    # generate image
    image_sim = imageModel.image(kwargs_macromodel_lens, kwargs_macromodel_source, kwargs_macromodel_lens_light)

    f, axes = plt.subplots(1, 1, figsize=(6, 6), sharex='none', sharey='none')
    ax = axes
    axes.matshow(np.log10(image_sim), origin='lower')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.autoscale(False)
    f.savefig(os.path.join(figure_dir, f'{data_set_name}_clean_model.png'))
    plt.close()

    execution_end_time = time.time()
    execution_time = execution_end_time - execution_start_time
    execution_times.append(execution_time)

np.save('modeling_pipeline_execution_times', execution_times)
