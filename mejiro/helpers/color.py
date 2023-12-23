from copy import deepcopy

import numpy as np
from astropy.visualization import make_lupton_rgb


# TODO some kind of wrapper to build color image given a lens class

# TODO method for linear color transformation

def get_rgb(image_b, image_g, image_r):
    # assert image_b.shape == image_g.shape == image_r.shape
    minimum = np.min(np.concatenate((image_b, image_g, image_r)))
    return make_lupton_rgb(image_r=image_r, image_g=image_g, image_b=image_b, minimum=minimum, stretch=3, Q=8)


def update_kwargs_magnitude(old_kwargs, new_magnitude):
    new_kwargs = deepcopy(old_kwargs)
    new_kwargs['magnitude'] = new_magnitude

    return new_kwargs
