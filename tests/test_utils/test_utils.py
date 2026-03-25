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


def test_smooth_negative_pixels():
    # no negative pixels: image is unchanged
    image = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])
    original = image.copy()
    util.smooth_negative_pixels(image)
    np.testing.assert_array_equal(image, original)

    # negative pixel is replaced with a non-negative value
    image = np.array([[1., 2., 3.],
                      [4., -5., 6.],
                      [7., 8., 9.]], dtype=float)
    util.smooth_negative_pixels(image)
    assert image[1, 1] >= 0

    # all non-negative pixels are preserved
    image = np.array([[1., 2., 3.],
                      [4., -5., 6.],
                      [7., 8., 9.]], dtype=float)
    util.smooth_negative_pixels(image)
    non_neg_mask = np.array([[True, True, True],
                             [True, False, True],
                             [True, True, True]])
    np.testing.assert_array_equal(image[non_neg_mask],
                                  np.array([1., 2., 3., 4., 6., 7., 8., 9.]))
    image = np.array([[1., -2.], [3., 4.]], dtype=float)
    util.smooth_negative_pixels(image)  # should not raise
    image = np.array([[1., 2.], [3., 4.]], dtype=float)
    util.smooth_negative_pixels(image)  # should not raise

    # in-place: returns the same array object
    image = np.array([[1., -2.], [3., 4.]], dtype=float)
    result = util.smooth_negative_pixels(image)
    assert result is image


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
