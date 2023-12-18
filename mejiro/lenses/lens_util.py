import pickle

import matplotlib.pyplot as plt
import numpy as np

from mejiro.lenses.lens import Lens


def unpickle_lens(pickle_path, uid):
    with open(pickle_path, 'rb') as pickled_lens:
        unpickled = pickle.load(pickled_lens)

    kwargs_model = unpickled['kwargs_model']
    kwargs_params = unpickled['kwargs_params']

    return Lens(kwargs_model=kwargs_model, kwargs_params=kwargs_params, uid=uid)


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
