import numpy as np
from scipy.optimize import curve_fit


import numpy as np
from scipy.optimize import curve_fit


def chi_square(a, b):
    
    """
    Compute the chi-square statistic between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        First input array. Must have the same shape as `b`.
    b : numpy.ndarray
        Second input array. Must have the same shape as `a`.

    Returns
    -------
    float
        The chi-square statistic. If any division by zero occurs, the result will be replaced with NaN.

    Raises
    ------
    AssertionError
        If the input arrays `a` and `b` do not have the same shape.
    AssertionError
        If the input array `b` contains any zeros.

    Notes
    -----
    If the input arrays are not 1-dimensional, they will be flattened before computation.
    """
    assert a.shape == b.shape, 'Arrays must have the same shape'
    assert np.count_nonzero(b) == len(b.flatten()), 'Array b has at least one element equal to zero'

    if a.ndim != 1:
        a = a.flatten()
        b = b.flatten()

    chi2 = 0
    for i, j in zip(a, b):
        chi2 += ((i - j) ** 2) / j
    
    return chi2


def normalize(array):
    """
    Normalize an array by dividing each element by the sum of all elements.

    Parameters
    ----------
    array : numpy.ndarray
        Input array to be normalized.

    Returns
    -------
    numpy.ndarray
        Normalized array where the sum of all elements is 1.
    """
    sum = np.sum(array)
    assert sum != 0, 'Sum of elements is zero'
    return array / sum


def linear_fit_through_origin(x, y):
    """
    Perform a linear fit through the origin.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable data.
    y : numpy.ndarray
        Dependent variable data.

    Returns
    -------
    x_fit : numpy.ndarray
        Fitted x values.
    y_fit : numpy.ndarray
        Fitted y values.
    c : float
        Slope of the fitted line.
    """
    def fit_func(x, c):
        return c * x

    params = curve_fit(fit_func, x, y)
    c = params[0][0]
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = c * x_fit

    return x_fit, y_fit, c
