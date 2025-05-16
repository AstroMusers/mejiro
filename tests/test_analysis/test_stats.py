import numpy as np
import pytest

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
