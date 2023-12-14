import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.fft import fft2

from mejiro.plots import plot_util


def residual_compare(ax, array_list, title_list, linear_width):
    norm = plot_util.get_norm(array_list, linear_width)

    for i, array in enumerate(array_list):
        axis = ax[i].imshow(array, cmap='bwr', norm=norm)
        ax[i].set_title(title_list[i])
        ax[i].set_axis_off()

    return axis


def fft(filepath, title, array):
    fft = fft2(array)
    plt.matshow(np.abs(fft), norm=colors.LogNorm())
    plt.title(title)
    plt.colorbar()
    plot_util.__savefig(filepath)
    plt.show()


def residual(array1, array2, title='', normalization=1):
    residual = (array1 - array2) / normalization
    abs_min, abs_max = abs(np.min(residual)), abs(np.max(residual))
    limit = np.max([abs_min, abs_max])
    linear_width = np.abs(np.mean(residual) + (3 * np.std(residual)))

    fig, ax = plt.subplots()
    im = ax.imshow(residual, cmap='bwr', norm=colors.AsinhNorm(linear_width=linear_width, vmin=-limit, vmax=limit))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    plt.show()


def execution_time_scatter(execution_times, title=''):
    plt.scatter(np.arange(0, len(execution_times)), execution_times)
    plt.title(title)

    plt.show()


def execution_time_hist(execution_times, title=''):
    plt.hist(execution_times)
    plt.title(title)

    plt.show()
