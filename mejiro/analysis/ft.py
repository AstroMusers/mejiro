import numpy as np


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


def get_k_list(min_arcsec, max_arcsec, num_pix):
    k_min = (2 * np.pi) / max_arcsec
    k_max = (2 * np.pi) / min_arcsec

    return np.linspace(k_min, k_max, num_pix, endpoint=True)


def get_theta_list(arcsec_min, arcsec_max, length):
    return np.linspace(arcsec_min, arcsec_max, length, endpoint=True)
