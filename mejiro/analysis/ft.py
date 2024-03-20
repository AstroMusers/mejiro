import numpy as np


def power_spectrum(image):
    assert image.shape[0] == image.shape[1], '2D array must be square'

    # get Fourier amplitudes
    fft = np.fft.fft2(image)
    fft_squared = np.square(np.abs(fft))

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


def get_k_list(min_arcsec, max_arcsec, num_points):
    k_min = (2 * np.pi) / max_arcsec
    k_max = (2 * np.pi) / min_arcsec

    return np.linspace(k_min, k_max, num_points, endpoint=True)


def get_theta_list(arcsec_min, arcsec_max, num_points):
    return np.linspace(arcsec_min, arcsec_max, num_points, endpoint=True)


def oversampled_power_spectrum(image, oversample, arcsec_min, arcsec_max):
    assert oversample % 2 != 0, 'Oversample factor must be odd'
    assert image.shape[0] == image.shape[1], '2D array must be square'

    # calculate theta list
    num_points = int(oversample * image.shape[0])
    theta_list = get_theta_list(arcsec_min, arcsec_max, num_points)

    oversampled_image = image.repeat(oversample, axis=0).repeat(oversample, axis=1)

    # get Fourier amplitudes
    fft = np.fft.fft2(oversampled_image)
    fft_squared = np.square(np.abs(fft))

    radius_list = range(oversampled_image.shape[0])

    # compute power in each radius of Fourier amplitudes array
    power_list = []
    for radius in radius_list:
        power_per_radius = []
        for x, row in enumerate(fft_squared):
            for y, _ in enumerate(row):
                if radius == round(np.sqrt((x ** 2) + (y ** 2))):
                    power_per_radius.append(fft_squared[x][y])

        power_list.append(np.sum(power_per_radius))

    return theta_list, power_list
