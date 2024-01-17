from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from mejiro.helpers import color

from mejiro.lenses.strong_lens import StrongLens
from mejiro.utils import util


def unpickle_lens(pickle_path, uid):
    unpickled = util.unpickle(pickle_path)

    kwargs_model = unpickled['kwargs_model']
    kwargs_params = unpickled['kwargs_params']
    lens_mags = unpickled['lens_mags']
    source_mags = unpickled['source_mags']

    return StrongLens(kwargs_model=kwargs_model, 
                      kwargs_params=kwargs_params, 
                      lens_mags=lens_mags, 
                      source_mags=source_mags,
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


def get_sample(pickle_dir, pandeia_dir, index):
    files = glob(pickle_dir + f'/lens_dict_{str(index).zfill(8)}_*')

    f106 = [util.unpickle(i) for i in files if 'f106' in i][0]
    f129 = [util.unpickle(i) for i in files if 'f129' in i][0]
    # f158 = [util.unpickle(i) for i in files if 'f158' in i][0]
    f184 = [util.unpickle(i) for i in files if 'f184' in i][0]

    rgb_model = color.get_rgb(f106['model'], f129['model'], f184['model'], minimum=None, stretch=3, Q=8)

    image_path = os.path.join(pandeia_dir, f'pandeia_color_{str(index).zfill(8)}.npy')
    rgb_image = np.load(image_path)

    return f106, rgb_image, rgb_model
