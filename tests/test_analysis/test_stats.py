import pytest
import numpy as np

from mejiro.analysis import stats

def test_chi_square():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    assert stats.chi_square(a, b) == 0

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [3, 4]])
    assert stats.chi_square(a, b) == 0

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 2], [3, 4]])
    assert stats.chi_square(a, b) == 0.5

    a = np.array([1, 2, 3])
    b = np.array([2, 3])
    with pytest.raises(AssertionError):
        stats.chi_square(a, b)

    a = np.array([1, 2, 3])
    b = np.array([0, 2, 3])
    with pytest.raises(AssertionError):
        stats.chi_square(a, b)


def test_normalize():
    array = np.array([1, 2, 3])
    normalized_array = stats.normalize(array)
    assert np.isclose(np.sum(normalized_array), 1)
    assert np.allclose(normalized_array, np.array([0.16666667, 0.33333333, 0.5]))

    array = np.array([0, 0, 0])
    with pytest.raises(AssertionError):
        stats.normalize(array)


def test_linear_fit_through_origin():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    x_fit, y_fit, c = stats.linear_fit_through_origin(x, y)
    assert np.isclose(c, 2)
    assert len(x_fit) == 100
    assert len(y_fit) == 100
    assert np.allclose(y_fit, 2 * x_fit)
