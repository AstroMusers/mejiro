import os
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util import util as lenstronomy_util

# use mejiro plotting style
import mejiro
from mejiro.helpers import color
from mejiro.lenses.strong_lens import StrongLens
from mejiro.utils import util

module_path = os.path.dirname(mejiro.__file__)
plt.style.use(f'{module_path}/mplstyle/science.mplstyle')


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
                    return True, None

    return False, None


def slsim_lens_to_mejiro(slsim_lens, bands, cosmo, snr=None, uid=None):
    kwargs_model, kwargs_params = slsim_lens.lenstronomy_kwargs(band=bands[0])

    lens_mags, source_mags = {}, {}
    for band in bands:
        lens_mags[band] = slsim_lens.deflector_magnitude(band)
        source_mags[band] = slsim_lens.extended_source_magnitude(band)

    z_lens, z_source = slsim_lens.deflector_redshift, slsim_lens.source_redshift
    kwargs_lens = kwargs_params['kwargs_lens']

    # add additional necessary key/value pairs to kwargs_model
    kwargs_model['lens_redshift_list'] = [z_lens] * len(kwargs_lens)
    kwargs_model['source_redshift_list'] = [z_source]
    kwargs_model['cosmo'] = cosmo
    kwargs_model['z_source'] = z_source
    kwargs_model['z_source_convention'] = 5

    return StrongLens(kwargs_model=kwargs_model,
                      kwargs_params=kwargs_params,
                      lens_mags=lens_mags,
                      source_mags=source_mags,
                      lens_stellar_mass=slsim_lens.deflector_stellar_mass(),
                      lens_vel_disp=slsim_lens.deflector_velocity_dispersion(),
                      snr=snr,
                      uid=uid)


def unpickle_lens(pickle_path, uid):
    unpickled = util.unpickle(pickle_path)

    kwargs_model = unpickled['kwargs_model']
    kwargs_params = unpickled['kwargs_params']
    lens_mags = unpickled['lens_mags']
    source_mags = unpickled['source_mags']
    lens_stellar_mass = unpickled['deflector_stellar_mass']
    lens_vel_disp = unpickled['deflector_velocity_dispersion']
    snr = unpickled['snr']

    return StrongLens(kwargs_model=kwargs_model,
                      kwargs_params=kwargs_params,
                      lens_mags=lens_mags,
                      source_mags=source_mags,
                      lens_stellar_mass=lens_stellar_mass,
                      lens_vel_disp=lens_vel_disp,
                      snr=snr,
                      uid=uid)


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


def get_sample(pickle_dir, color_dir, index):
    # get lens
    lens_path = os.path.join(pickle_dir, f'lens_{str(index).zfill(8)}.pkl')
    lens = util.unpickle(lens_path)

    # get rgb model
    files = glob(pickle_dir + f'/array_{str(index).zfill(8)}_*')
    f106 = [np.load(i) for i in files if 'F106' in i][0]
    f129 = [np.load(i) for i in files if 'F129' in i][0]
    # f158 = [np.load(i) for i in files if 'F158' in i][0]
    f184 = [np.load(i) for i in files if 'F184' in i][0]
    rgb_model = color.get_rgb(f106, f129, f184, minimum=None, stretch=3, Q=8)

    # get rgb image
    image_path = os.path.join(color_dir, f'galsim_color_{str(index).zfill(8)}.npy')
    rgb_image = np.load(image_path)

    return lens, rgb_model, rgb_image


def update_kwargs_magnitude(old_kwargs, new_magnitude):
    new_kwargs = deepcopy(old_kwargs)
    new_kwargs['magnitude'] = new_magnitude

    return new_kwargs
