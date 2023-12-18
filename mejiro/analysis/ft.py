import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def twod_ft(array):
    fourier_image = np.fft.fftn(array)
    fourier_amplitudes = np.abs(fourier_image)**2
    return fourier_amplitudes


# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
def power_spectrum(array, k_min=1, k_max=22):
    npix = array.shape[0]

    fourier_image = np.fft.fftn(array)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    kvals = np.linspace(k_min, k_max, 22)

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins


# https://github.com/franyancr/Subhalo-correlations/blob/master/plot_power_spectrum.py
def twoD_ps(data=None, pix_size=0, rnge=100, shift=0, show_ps=False):
    """
    takes in a 2D array and returns the 2D FFT:
    inputs:
        ind_coords: n 2d arrays, whose average = coadd_coords
        coadd_coords: a singe 2d array
        pix_size: pixel size
        rnge: box size
        shift: offset from the halo center
        show_ps: whether you want to display the 2d power spectrum
    outputs:
        ind_ps_x: list of lists, where each list is a 2d power spectrum
        tot_ps: total 2D power spectrum after coadding all PS in ind_ps_x
        ky,ky: fft frequencies
    """
    if data is None:
        raise RuntimeError('No data was given!')

    ind_ps = []
    for i in data:
        ft = np.fft.fft2(i)
        ps2D = np.abs(ft) ** 2
        ind_ps.append(ps2D)

    A_pix = pix_size ** 2
    A_box = rnge ** 2
    norm = A_pix ** 2 / A_box

    ind_ps_x = [norm * np.fft.fftshift(i) for i in ind_ps]
    tot_ps = np.mean(ind_ps_x, axis=0)
    tot_ps2 = np.log10(tot_ps)

    kx = 2 * np.pi * np.fft.fftfreq(tot_ps.shape[0], d=pix_size)
    kx = np.fft.fftshift(kx)
    ky = 2 * np.pi * np.fft.fftfreq(tot_ps.shape[1], d=pix_size)
    ky = np.fft.fftshift(ky)

    if show_ps == True:
        fig, (ax1, ax2) = plt.subplots(2, sharey=True)
        ax1.imshow(tot_ps, extent=[min(kx), max(kx), min(ky), max(ky)], interpolation='nearest')
        ax2.imshow(tot_ps2, extent=[min(kx), max(kx), min(ky), max(ky)], interpolation='nearest')
        plt.show()

    return ind_ps_x, tot_ps, kx, ky
