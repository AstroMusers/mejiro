import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from matplotlib import rc
from scipy.fft import fft2

from package.plots import plot_util

rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace': ['Computer Modern Typewriter']})
rc('text', usetex=True)

matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'


# TODO execution time-dependence of whatever other parameters I can vary


def fft(filepath, title, array):
    fft = fft2(array)
    plt.matshow(np.abs(fft), norm=colors.LogNorm())
    plt.title(title)
    plt.colorbar()
    plot_util.__savefig(filepath)
    plt.show()


def residual(filepath, title, array1, array2, normalization=1):
    residual = (array1 - array2) / normalization
    abs_min, abs_max = abs(np.min(residual)), abs(np.max(residual))
    limit = np.max([abs_min, abs_max])
    linear_width = np.abs(np.mean(residual) + (3 * np.std(residual)))

    fig, ax = plt.subplots()
    im = ax.imshow(residual, cmap='bwr', norm=colors.AsinhNorm(linear_width=linear_width, vmin=-limit, vmax=limit))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
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
