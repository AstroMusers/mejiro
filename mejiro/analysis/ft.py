import Pk_library as PKL
import numpy as np
import scipy.stats as stats


def power_spectrum(image):
    # get Fourier amplitudes
    fft = np.fft.fft2(image)
    fft_squared = np.square(np.abs(fft))

    assert image.shape[0] == image.shape[1], '2D array must be square'
    radius_list = range(image.shape[0])

    # compute power in each radius of Fourier amplitudes array
    power_list = []
    for radius in radius_list:
        power_per_radius = []
        for x, row in enumerate(fft_squared):
            for y, _ in enumerate(row):
                if radius == round(np.sqrt((x ** 2) + (y ** 2))):
                    power_per_radius.append(fft_squared[x][y])

        power_list.append(np.sum(power_per_radius))

    return power_list


def get_k_list(min_arcsec, max_arcsec, length):
    k_min = (2 * np.pi) / max_arcsec
    k_max = (2 * np.pi) / min_arcsec

    return np.linspace(k_min, k_max, length, endpoint=True)


def get_theta_list(arcsec_min, arcsec_max, length):
    return np.linspace(arcsec_min, arcsec_max, length, endpoint=True)


# TODO delete, once my method finalized
def twod_ft(array, box_size, threads=1):
    # compute the Pk of that image
    Pk2D = PKL.Pk_plane(array.astype(dtype=np.float32), box_size, 'None', threads)

    # Pk2D.k: k in 1/arcsec
    # Pk2D.Pk: Pk in (arcsec)^2
    # Pk2D.Nmodes: Number of modes in the different k bins

    return Pk2D.k, Pk2D.Pk, Pk2D.Nmodes


# TODO delete, once my method finalized
# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
# def power_spectrum(array, k_min=1, k_max=22):
#     side = array.shape[0]

#     # take Fourier transform
#     fourier_image = np.fft.fft2(array)

#     # get variances
#     fourier_amplitudes = np.abs(fourier_image) ** 2

#     # get wave vectors in units of inverse pixels
#     kfreq = np.fft.fftfreq(side) * side

#     # make grid of the norm of these wave vectors
#     kfreq2D = np.meshgrid(kfreq, kfreq)
#     knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

#     # flatten so we can plot in one dimension
#     knrm = knrm.flatten()
#     fourier_amplitudes = fourier_amplitudes.flatten()

#     kbins = np.arange(0.5, side // 2 + 1, 1.)  # can only go up to N/2 reliably?
#     kvals = 0.5 * (kbins[1:] + kbins[:-1])
#     # kvals = np.linspace(k_min, k_max, 22)

#     average_power, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
#                                                  statistic="mean",
#                                                  bins=kbins)
#     total_power = average_power * np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

#     return kvals, total_power
