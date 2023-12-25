from copy import deepcopy

import numpy as np
from astropy.visualization import make_lupton_rgb


def get_rgb(image_b, image_g, image_r, minimum=None, stretch=3, Q=4):
    # assert image_b.shape == image_g.shape == image_r.shape
    if minimum is None:
        minimum = np.min(np.concatenate((image_b, image_g, image_r)))
    return make_lupton_rgb(image_r=image_r, image_g=image_g, image_b=image_b, minimum=minimum, stretch=stretch, Q=Q)


def get_rgb_log10(image_b, image_g, image_r):
    image = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
    image[:,:,0] = np.log10(image_b)
    image[:,:,1] = np.log10(image_g)
    image[:,:,2] = np.log10(image_r)
    return image


def update_kwargs_magnitude(old_kwargs, new_magnitude):
    new_kwargs = deepcopy(old_kwargs)
    new_kwargs['magnitude'] = new_magnitude

    return new_kwargs
