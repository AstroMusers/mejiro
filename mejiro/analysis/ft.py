import Pk_library as PKL
import numpy as np
import scipy.stats as stats


def twod_ft(array, box_size, threads=1):
    # compute the Pk of that image
    Pk2D = PKL.Pk_plane(array.astype(dtype=np.float32), box_size, 'None', threads)

    # Pk2D.k: k in 1/arcsec
    # Pk2D.Pk: Pk in (arcsec)^2
    # Pk2D.Nmodes: Number of modes in the different k bins

    return Pk2D.k, Pk2D.Pk, Pk2D.Nmodes


# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
def power_spectrum(array, k_min=1, k_max=22):
    side = array.shape[0]

    # take Fourier transform
    fourier_image = np.fft.fft2(array)

    # get variances
    fourier_amplitudes = np.abs(fourier_image) ** 2

    # get wave vectors in units of inverse pixels
    kfreq = np.fft.fftfreq(side) * side

    # make grid of the norm of these wave vectors
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    # flatten so we can plot in one dimension
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, side // 2 + 1, 1.)  # can only go up to N/2 reliably?
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # kvals = np.linspace(k_min, k_max, 22)

    average_power, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                                 statistic="mean",
                                                 bins=kbins)
    total_power = average_power * np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    return kvals, total_power
