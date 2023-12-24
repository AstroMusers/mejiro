from copy import deepcopy

import numpy as np
from astropy.visualization import make_lupton_rgb


def get_rgb(image_b, image_g, image_r, minimum=None, stretch=3, Q=8):
    # assert image_b.shape == image_g.shape == image_r.shape
    if minimum is None:
        minimum = np.min(np.concatenate((image_b, image_g, image_r)))
    return make_lupton_rgb(image_r=image_r, image_g=image_g, image_b=image_b, minimum=minimum, stretch=stretch, Q=Q)


def update_kwargs_magnitude(old_kwargs, new_magnitude):
    new_kwargs = deepcopy(old_kwargs)
    new_kwargs['magnitude'] = new_magnitude

    return new_kwargs
