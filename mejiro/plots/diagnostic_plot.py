import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.fft import fft2

from mejiro.lenses import lens_util
from mejiro.plots import plot_util


def snr_plot(labeled_array, strong_lens, total, lens, source, noise, snr_array, masked_snr_array, snr_list, debug_dir):
    _, ax = plt.subplots(2, 3, figsize=(12, 8))

    # vmin, vmax = plot_util.get_min_max([total, lens, source, noise])

    lens_mag = strong_lens.lens_mags['F129']
    source_mag = strong_lens.source_mags['F129']

    im00 = ax[0][0].imshow(np.log10(total))  # , vmin=vmin, vmax=vmax
    plt.colorbar(im00, ax=ax[0][0])
    ax[0][0].set_title('Total Image (log10)')

    im01 = ax[0][1].imshow(lens)
    plt.colorbar(im01, ax=ax[0][1])
    ax[0][1].set_title('Lens (' + r'$m_\textrm{F129}=$' + f'{lens_mag:.2f})')

    im02 = ax[0][2].imshow(source)
    plt.colorbar(im02, ax=ax[0][2])
    ax[0][2].set_title('Source (' + r'$m_\textrm{F129}=$' + f'{source_mag:.2f})')

    im10 = ax[1][0].imshow(noise)
    plt.colorbar(im10, ax=ax[1][0])
    ax[1][0].set_title('Noise')

    im11 = ax[1][1].imshow(labeled_array)
    # for k, region in enumerate(indices_list):
    #     for i, j in region:
    #         ax[1][1].plot(j, i, 'ro', markersize=1, color=f'C{k}')
    plt.colorbar(im11, ax=ax[1][1])
    ax[1][1].set_title('Labeled Array')

    im12 = ax[1][2].imshow(masked_snr_array)
    plt.colorbar(im12, ax=ax[1][2])
    ax[1][2].set_title('Masked SNR Array')

    plt.suptitle(f'SNR: {np.max(snr_list)}, z_l={strong_lens.z_lens:.2f}, z_s={strong_lens.z_source:.2f}')
    try:
        plt.savefig(f'{debug_dir}/snr/snr_check_{id(total)}.png')
        plt.close()
    except:
        print('Could not save SNR plot')


def power_spectrum_check(array_list, lenses, titles, save_path, oversampled):
    if type(array_list[0]) is not np.ndarray:
        array_list = [i.array for i in array_list]

    f, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i, array in enumerate(array_list):
        axis = ax[0][i].imshow(np.log10(array))
        ax[0][i].set_title(titles[i])
        ax[0][i].axis('off')

    cbar = f.colorbar(axis, ax=ax[0])
    cbar.set_label('log(Counts)', rotation=90)

    res_array = [array_list[3] - array_list[i] for i in range(4)]
    v = plot_util.get_v(res_array)
    for i in range(4):
        axis = ax[1][i].imshow(array_list[3] - array_list[i], cmap='bwr', vmin=-v, vmax=v)
        ax[1][i].set_axis_off()

    cbar = f.colorbar(axis, ax=ax[1])
    cbar.set_label('Counts', rotation=90)

    for i, lens in enumerate(lenses):
        realization = lens.realization
        if realization is not None:
            for halo in realization.halos:
                if halo.mass > 1e8:
                    if oversampled:
                        coords = lens_util.get_coords(45 * 5, delta_pix=0.11 / 5)
                    else:
                        coords = lens_util.get_coords(45, delta_pix=0.11)
                    ax[1][i].scatter(*coords.map_coord2pix(halo.x, halo.y), s=100, facecolors='none',
                                     edgecolors='black')

    plt.savefig(save_path)
    plt.close()


def residual_compare(ax, array_list, linear_width, title_list=None):
    norm = plot_util.get_norm(array_list, linear_width)

    last_array = array_list[:-1]

    for i, array in enumerate(array_list):
        axis = ax[i].imshow(last_array - array, cmap='bwr', norm=norm)
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
