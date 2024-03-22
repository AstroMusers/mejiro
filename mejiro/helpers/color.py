import numpy as np
from astropy.visualization import make_lupton_rgb

from mejiro.plots import plot_util


def get_rgb(image_b, image_g, image_r, minimum=None, stretch=3, Q=4):
    # image_b, image_g, image_r = _rescale_rgb_float(image_b, image_g, image_r)

    # assert image_b.shape == image_g.shape == image_r.shape
    if minimum is None:
        min_r = np.min(image_r)
        min_g = np.min(image_g)
        min_b = np.min(image_b)
        # minimum = np.min(np.concatenate((image_b, image_g, image_r)))
        minimum = [min_r, min_g, min_b]
    return make_lupton_rgb(image_r=image_r, image_g=image_g, image_b=image_b, minimum=minimum, stretch=stretch, Q=Q)


def get_rgb_log10(image_b, image_g, image_r):
    image = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)

    image[:, :, 0] = np.log10(image_b)
    image[:, :, 1] = np.log10(image_g)
    image[:, :, 2] = np.log10(image_r)
    return image


def _rescale_rgb_float(image_b, image_g, image_r):
    max = plot_util.get_v([image_b, image_g, image_r])
    image_b /= max
    image_g /= max
    image_r /= max

    return image_b, image_g, image_r
