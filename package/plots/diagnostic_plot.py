import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
from scipy.fft import fft2

from package.plots import __savefig

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'


def fft(filepath, title, array):
    fft = fft2(array)
    plt.matshow(np.abs(fft), norm=matplotlib.colors.LogNorm())
    plt.title(title)
    plt.colorbar()
    __savefig(filepath)
    plt.show()


def residual(filepath, title, array1, array2):
    residual = (array1 - array2) / array1
    plt.imshow(residual, cmap='bwr')
    plt.title(title)
    plt.colorbar()
    __savefig(filepath)
    plt.show()


def execution_time(filepath, title, execution_times):
    plt.hist(np.arange(0, len(execution_times)), execution_times)
    plt.title(title)
    __savefig(filepath)
    plt.show()
