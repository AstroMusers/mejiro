import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
from scipy.fft import fft2

from package.plots import plot_util

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Source Sans Pro']})
rc('text', usetex=True)

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'


# TODO execution time-dependence of whatever other parameters I can vary


def fft(filepath, title, array):
    fft = fft2(array)
    plt.matshow(np.abs(fft), norm=matplotlib.colors.LogNorm())
    plt.title(title)
    plt.colorbar()
    plot_util.__savefig(filepath)
    plt.show()


def residual(filepath, title, array1, array2, normalization=1):
    residual = (array1 - array2) / normalization
    abs_min, abs_max = abs(np.min(residual)), abs(np.max(residual))
    limit = np.max([abs_min, abs_max])
    plt.imshow(residual, cmap='bwr', vmin=-limit, vmax=limit)
    plt.title(title)
    plt.colorbar()
    plot_util.__savefig(filepath)
    plt.show()


def execution_time_scatter(filepath, title, execution_times):
    plt.scatter(np.arange(0, len(execution_times)), execution_times)
    plt.title(title)
    plot_util.__savefig(filepath)
    plt.show()


def execution_time_hist(filepath, title, execution_times):
    plt.hist(execution_times)
    plt.title(title)
    plot_util.__savefig(filepath)
    plt.show()
