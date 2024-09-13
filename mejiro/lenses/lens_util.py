import os
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util import util as lenstronomy_util
from tqdm import tqdm

# use mejiro plotting style
import mejiro
from mejiro.helpers import color
from mejiro.lenses.strong_lens import StrongLens
from mejiro.utils import util


module_path = os.path.dirname(mejiro.__file__)
plt.style.use(f'{module_path}/mplstyle/science.mplstyle')


def count_detectable_lenses(dir):
    lens_pickles = glob(dir + '/**/detectable_lens_*.pkl')
    return len(lens_pickles)


def get_detectable_lenses(pipeline_dir, limit=None, with_subhalos=False, suppress_output=True):
    lens_list = []

    if with_subhalos:
        pickles = glob(os.path.join(pipeline_dir, '02', '**', 'lens_with_subhalos_*.pkl'))
        if limit is not None:
            pickles = np.random.choice(pickles, limit)
        for pickle in tqdm(pickles, disable=suppress_output):
            lens_list.append(util.unpickle(pickle))
    else:
        pickles = glob(os.path.join(pipeline_dir, '01', '01_hlwas_sim_detectable_lenses_sca*.pkl'))
        if limit is not None:
            pickles = np.random.choice(pickles, limit)
        for pickle in tqdm(pickles, disable=suppress_output):
            lens_list.extend(util.unpickle(pickle))

    assert len(lens_list) != 0, f'No pickled lenses found. Check {pipeline_dir}.'

    return lens_list


def overplot_subhalos(lens, num_pix=91, side=10.01, band='F106', figsize=7):
    # make sure there are subhalos on this StrongLens
    if lens.realization is None:
        raise ValueError('No subhalos have been added to this StrongLens object.')

    # get array
    array = lens.get_array(num_pix, side, band=band)

    # plot
    f = plt.figure(figsize=(figsize, figsize))
    ax = plt.subplot(111)
    ax.imshow(np.log10(array))

    # TODO make sure this method can handle different oversampling factors

    # overplot subhalos
    coords = get_coords(num_pix, delta_pix=0.11)

    for halo in lens.realization.halos:
        if halo.mass > 1e8:
            ax.plot(*coords.map_coord2pix(halo.x, halo.y), marker='.', color='#FF9500')
        elif halo.mass > 1e7:
            ax.plot(*coords.map_coord2pix(halo.x, halo.y), marker='.', color='#00B945')
        else:
            ax.plot(*coords.map_coord2pix(halo.x, halo.y), marker='.', color='#0C5DA5')

    plt.show()


def get_coords(num_pix, delta_pix=0.11, subgrid_res=1):
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, Mcoord2pix = lenstronomy_util.make_grid_with_coordtransform(
        numPix=num_pix,
        deltapix=delta_pix,
        subgrid_res=subgrid_res,
        left_lower=False,
        inverse=False)

    return Coordinates(Mpix2coord, ra_at_xy_0, dec_at_xy_0)


def check_halo_image_alignment(lens, realization, halo_mass=1e8, halo_sort_massive_first=True, return_halo=False):
    sorted_halos = sorted(realization.halos, key=lambda x: x.mass, reverse=halo_sort_massive_first)

    # get image position
    source_x = lens.kwargs_source_dict['F106']['center_x']
    source_y = lens.kwargs_source_dict['F106']['center_y']
    solver = LensEquationSolver(lens.lens_model_class)
    image_x, image_y = solver.image_position_from_source(sourcePos_x=source_x, sourcePos_y=source_y,
                                                         kwargs_lens=lens.kwargs_lens)

    for halo in sorted_halos:
        if halo.mass < halo_mass:
            break

        # calculate distances
        for x, y in zip(image_x, image_y):
            dist = np.sqrt(np.power(halo.x - x, 2) + np.power(halo.y - y, 2))

            # check if halo is within 0.1 arcsec of the image
            if dist < 0.1:
                if return_halo:
                    return True, halo
                else:
                    return True
    if return_halo:
        return False, None
    else:
        return False


def slsim_lens_to_mejiro(slsim_lens, bands, cosmo, snr=None, uid=None, z_source_convention=6, sca=None):
    kwargs_model, kwargs_params = slsim_lens.lenstronomy_kwargs(band=bands[0])

    lens_mags, source_mags, lensed_source_mags = {}, {}, {}
    for band in bands:
        lens_mags[band] = slsim_lens.deflector_magnitude(band)
        source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=False)
        lensed_source_mags[band] = slsim_lens.extended_source_magnitude(band, lensed=True)

    z_lens, z_source = slsim_lens.deflector_redshift, slsim_lens.source_redshift
    kwargs_lens = kwargs_params['kwargs_lens']

    # add additional necessary key/value pairs to kwargs_model
    kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
    kwargs_model['source_redshift_list'] = [z_source]
    kwargs_model['cosmo'] = cosmo
    kwargs_model['z_source'] = z_source
    kwargs_model['z_source_convention'] = z_source_convention

    return StrongLens(kwargs_model=kwargs_model,
                      kwargs_params=kwargs_params,
                      lens_mags=lens_mags,
                      source_mags=source_mags,
                      lensed_source_mags=lensed_source_mags,
                      lens_stellar_mass=slsim_lens.deflector_stellar_mass(),
                      lens_vel_disp=slsim_lens.deflector_velocity_dispersion(),
                      magnification=slsim_lens.extended_source_magnification(),
                      snr=snr,
                      uid=uid,
                      sca=sca)


def unpickle_lens(pickle_path, uid):
    unpickled = util.unpickle(pickle_path)

    kwargs_model = unpickled['kwargs_model']
    kwargs_params = unpickled['kwargs_params']
    lens_mags = unpickled['lens_mags']
    source_mags = unpickled['source_mags']
    lensed_source_mags = unpickled['lensed_source_mags']
    lens_stellar_mass = unpickled['deflector_stellar_mass']
    lens_vel_disp = unpickled['deflector_velocity_dispersion']
    magnification = unpickled['magnification']
    snr = unpickled['snr']
    masked_snr_array = unpickled['masked_snr_array']
    sca = unpickled['sca']

    return StrongLens(kwargs_model=kwargs_model,
                      kwargs_params=kwargs_params,
                      lens_mags=lens_mags,
                      source_mags=source_mags,
                      lensed_source_mags=lensed_source_mags,
                      lens_stellar_mass=lens_stellar_mass,
                      lens_vel_disp=lens_vel_disp,
                      magnification=magnification,
                      snr=snr,
                      masked_snr_array=masked_snr_array,
                      uid=uid,
                      sca=sca)


# TODO check references and fix
def set_kwargs_params(kwargs_lens, kwargs_lens_light, kwargs_source):
    return {
        'kwargs_lens': kwargs_lens,
        'kwargs_lens_light': kwargs_lens_light,
        'kwargs_source': kwargs_source
    }


def set_kwargs_model(lens_model_list, lens_light_model_list, source_model_list):
    return {
        'lens_model_list': lens_model_list,
        'lens_light_model_list': lens_light_model_list,
        'source_light_model_list': source_model_list
    }


def plot_projected_mass(lens):
    npix = 100
    _x = _y = np.linspace(-1.2, 1.2, npix)
    xx, yy = np.meshgrid(_x, _y)
    shape0 = xx.shape
    kappa_subs = lens.lens_model_class.kappa(xx.ravel(), yy.ravel(), lens.kwargs_lens).reshape(shape0)

    _, ax = plt.figure()
    return ax.imshow(kappa_subs, vmin=-0.1, vmax=0.1, cmap='bwr')


def get_sample(pipeline_dir, index, band=None, rgb_bands=['F184', 'F129', 'F106'], model=True, model_stretch=2, model_Q=3):
    # get lens
    lens_dir = pipeline_dir + '/03'
    lens_path = glob(lens_dir + f'/**/lens_{str(index).zfill(8)}.pkl')
    assert len(lens_path) == 1, f'StrongLens {index} not found in {lens_dir}.'
    lens = util.unpickle(lens_path[0])

    if model:
        model_dir = pipeline_dir + '/03'
        files = glob(model_dir + f'/**/array_{str(index).zfill(8)}_*.npy')
        assert len(files) != 0, f'Synthetic image files for StrongLens {index} not found in {model_dir}.'
    else:
        image_dir = pipeline_dir + '/04'
        files = glob(image_dir + f'/**/galsim_{str(index).zfill(8)}_*.npy')
        assert len(files) != 0, f'Exposure files for StrongLens {index} not found in {image_dir}.'

    r = [np.load(i) for i in files if rgb_bands[0] in i][0]
    g = [np.load(i) for i in files if rgb_bands[1] in i][0]
    b = [np.load(i) for i in files if rgb_bands[2] in i][0]
    rgb_model = color.get_rgb(r, g, b, minimum=None, stretch=model_stretch, Q=model_Q)

    # get rgb image
    color_dir = pipeline_dir + '/05'
    image_path = glob(color_dir + f'/**/galsim_color_{str(index).zfill(8)}.npy')
    assert len(image_path) == 1, f'Color image for StrongLens {index} not found in {color_dir}.'
    rgb_image = np.load(image_path[0])

    # TODO should be able to do something with the band arg
    return lens, rgb_model, rgb_image


def update_kwargs_magnitude(old_kwargs, new_magnitude):
    new_kwargs = deepcopy(old_kwargs)
    new_kwargs['magnitude'] = new_magnitude

    return new_kwargs
