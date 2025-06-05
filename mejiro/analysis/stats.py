import numpy as np
from scipy.optimize import curve_fit


def chi_square(a, b):
    """
    Compute the chi-square statistic between two arrays:

    .. math::

        \\chi^2 = \\sum_{i} \\frac{\\left(A_{i} - B_{i}\\right)^2}{B_{i}}

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
