import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

from package.plots import __savefig

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'


# TODO add method 98th percentile linear scaling


def log10(filepath, title, array, cmap='viridis', colorbar=False):
    plt.imshow(np.log10(array), cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    __savefig(filepath)
    plt.show()


def plot(filepath, title, array, cmap='viridis', colorbar=False):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    __savefig(filepath)
    plt.show()
