import numpy as np
from scipy.optimize import curve_fit


def chi_square(a, b):
    assert a.shape == b.shape, 'Arrays must have the same shape'

    if a.ndim != 1:
        a = a.flatten()
        b = b.flatten()

    chi2 = 0
    for i, j in zip(a, b):
        chi2 += ((i - j) ** 2) / j
    
    return np.nan_to_num(chi2, copy=False)


def chi2_distance(A, B):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(A, B)])
    return chi


def normalize(array):
    sum = np.sum(array)
    return array / sum


def linear_fit_through_origin(x, y):
    def fit_func(x, c):
        return c * x

    params = curve_fit(fit_func, x, y)
    c = params[0][0]
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = c * x_fit

    return x_fit, y_fit, c
