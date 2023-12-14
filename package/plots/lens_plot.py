import matplotlib.pyplot as plt
import numpy as np


def plot_projected_mass(lens):
    npix = 100
    _x = _y = np.linspace(-1.2, 1.2, npix)
    xx, yy = np.meshgrid(_x, _y)
    shape0 = xx.shape
    kappa_subs = lens.lens_model_class.kappa(xx.ravel(), yy.ravel(), lens.kwargs_lens).reshape(shape0)

    _, ax = plt.figure()
    return ax.imshow(kappa_subs, vmin=-0.1, vmax=0.1, cmap='bwr')
