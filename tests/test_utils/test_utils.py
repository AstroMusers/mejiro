import numpy as np
import pytest

from mejiro.utils import util


def test_smallest_non_negative_element():
    # Test with an array containing both negative and non-negative elements
    array = np.array([-1, 0, 2, -3, 5])
    assert util.smallest_non_negative_element(array) == 0

    # Test with an array containing only non-negative elements
    array = np.array([3, 1, 4, 2])
    assert util.smallest_non_negative_element(array) == 1

    # Test with an array containing only negative elements
    array = np.array([-1, -2, -3])
    assert util.smallest_non_negative_element(array) is None

    # Test with an empty array
    array = np.array([])
    assert util.smallest_non_negative_element(array) is None

    # Test with an array containing a single non-negative element
    array = np.array([5])
    assert util.smallest_non_negative_element(array) == 5

    # Test with an array containing a single negative element
    array = np.array([-5])
    assert util.smallest_non_negative_element(array) is None

    # Test with an array containing zero
    array = np.array([0, 1, 2])
    assert util.smallest_non_negative_element(array) == 0

    # Test with an array containing large numbers
    array = np.array([1e10, 1e5, 1e2])
    assert util.smallest_non_negative_element(array) == 100.0


def test_replace_negatives():
    # Test with an array containing negative values
    arr = np.array([-1, 2, -3, 4])
    with pytest.warns(UserWarning,
                      match='Negative values in array have been replaced with 0'):
        result = util.replace_negatives(arr)
    expected = np.array([0, 2, 0, 4])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # Test with an array containing no negative values
    arr = np.array([1, 2, 3, 4])
    result = util.replace_negatives(arr)
    expected = np.array([1, 2, 3, 4])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # Test with an array containing all negative values
    arr = np.array([-1, -2, -3, -4])
    with pytest.warns(UserWarning,
                      match='Negative values in array have been replaced with 0'):
        result = util.replace_negatives(arr)
    expected = np.array([0, 0, 0, 0])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # Test with an empty array
    arr = np.array([])
    result = util.replace_negatives(arr)
    expected = np.array([])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # Test with a custom replacement value
    arr = np.array([-1, 2, -3, 4])
    with pytest.warns(UserWarning,
                      match='Negative values in array have been replaced with 99'):
        result = util.replace_negatives(arr, replacement=99)
    expected = np.array([99, 2, 99, 4])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # Test with a 2D array
    arr = np.array([[-1, 2], [-3, 4]])
    with pytest.warns(UserWarning,
                      match='Negative values in array have been replaced with 0'):
        result = util.replace_negatives(arr)
    expected = np.array([[0, 2], [0, 4]])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_create_centered_box():
    # check even N
    with pytest.raises(ValueError):
        util.create_centered_box(4, 4)

    # check even box size
    with pytest.raises(ValueError):
        util.create_centered_box(5, 4)

    # check box size \leq N
    with pytest.raises(ValueError):
        util.create_centered_box(5, 6)

    # check single True
    result = util.create_centered_box(1, 1)
    expected = np.ones((1, 1), dtype=bool)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # check all Trues
    result = util.create_centered_box(5, 5)
    expected = np.ones((5, 5), dtype=bool)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    # check a standard case
    result = util.create_centered_box(5, 3)
    expected = np.array([[False, False, False, False, False],
                         [False, True, True, True, False],
                         [False, True, True, True, False],
                         [False, True, True, True, False],
                         [False, False, False, False, False]])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"
