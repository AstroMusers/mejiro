import numpy as np
from scipy.optimize import curve_fit


def linear_fit_through_origin(x, y):
    def fit_func(x, c):
        return c * x

    params = curve_fit(fit_func, x, y)
    c = params[0][0]
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = c * x_fit

    return x_fit, y_fit, c
