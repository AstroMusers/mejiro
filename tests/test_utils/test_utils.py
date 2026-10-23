import numpy as np
import pytest

from mejiro.utils import util


@pytest.mark.parametrize('typeplotback', ['white', 'dark', 'black', 'norm'])
def test_aas_style_renders_requested_theme(typeplotback, tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from mejiro.style import set_aas_style
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style, save_figure

    with plot_style('white'):
        set_aas_style(typeplotback)
        figure, axis = plt.subplots()
        line, = axis.plot([0, 1], [1, 0], color='red', label='data')
        legend = axis.legend()
        FigureCanvasAgg(figure).draw()
        background, foreground = plot_background_colors(typeplotback)
        assert figure.get_facecolor() == to_rgba(background)
        assert axis.get_facecolor() == to_rgba(background)
        assert all(to_rgba(tick.get_color()) == to_rgba(foreground) for tick in axis.get_xticklabels())
        assert to_rgba(legend.get_texts()[0].get_color()) == to_rgba(foreground)
        assert plt.rcParams['font.size'] == 10
        assert line.get_color() == 'red'
        path = save_figure(figure, tmp_path / 'theme')
        with Image.open(path) as image:
            assert image.convert('RGB').getpixel((0, 0)) == tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        plt.close(figure)


@pytest.mark.parametrize("fwhm", [2.0, 6.0, 10.0])
def test_gaussian_kernel_preserves_original_width_and_boundary_mode(fwhm):
    from scipy.ndimage import gaussian_filter

    impulse = np.zeros((31, 31))
    impulse[15, 15] = 1.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # [pixel]
    expected = gaussian_filter(impulse, sigma=sigma)
    np.testing.assert_allclose(util.get_gaussian_kernel(fwhm, size=31), expected, rtol=1e-14)


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


@pytest.mark.parametrize(
    "legacy_helper",
    [util.smooth_negative_pixels, util.smooth_nan_pixels, util.smooth_pixels],
)
def test_legacy_smoothing_helpers_preserve_valid_pixels(legacy_helper):
    image = np.array([[0.0, 1.0], [2.0, 3.0]])
    original = image.copy()

    result = legacy_helper(image)

    assert result is image
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "legacy_helper",
    [util.smooth_negative_pixels, util.smooth_nan_pixels, util.smooth_pixels],
)
@pytest.mark.parametrize("bad_value", [-1.0, np.nan, np.inf])
def test_legacy_smoothing_helpers_reject_invalid_pixels_without_modifying(
    legacy_helper, bad_value
):
    image = np.array([[0.0, bad_value], [2.0, 3.0]])
    original = image.copy()

    with pytest.raises(ValueError, match="no pixels were replaced"):
        legacy_helper(image)

    np.testing.assert_array_equal(image, original)


def test_validate_rendered_image_preserves_valid_pixels():
    image = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    original = image.copy()

    result = util.validate_rendered_image(image, label="test scene")

    assert result is image
    np.testing.assert_array_equal(image, original)


def test_validate_rendered_image_preserves_input_dtype():
    image = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    result = util.validate_rendered_image(image)

    assert result is image
    assert result.dtype == image.dtype


@pytest.mark.parametrize("bad_value", [-1.0, np.nan, np.inf])
def test_validate_rendered_image_rejects_invalid_pixels(bad_value):
    image = np.array([[0.0, bad_value], [2.0, 3.0]])

    with pytest.raises(ValueError, match="no pixels were replaced"):
        util.validate_rendered_image(image, label="test scene")


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


def test_get_gaussian_kernel():
    # shape
    kernel = util.get_gaussian_kernel(fwhm=2.0, size=11)
    assert kernel.shape == (11, 11)

    # gaussian_filter on a unit impulse preserves total mass
    assert kernel.sum() == pytest.approx(1.0, abs=1e-12)

    # peak is at the center pixel
    center = (11 // 2, 11 // 2)
    assert np.unravel_index(np.argmax(kernel), kernel.shape) == center

    # 4-fold symmetry: horizontal, vertical, both diagonals
    np.testing.assert_allclose(kernel, kernel[::-1, :])
    np.testing.assert_allclose(kernel, kernel[:, ::-1])
    np.testing.assert_allclose(kernel, kernel.T)

    # FWHM check: at the half-width of fwhm/2 from center, value is half the peak.
    # use a large grid + integer fwhm so the half-width lands exactly on a pixel.
    big = util.get_gaussian_kernel(fwhm=10.0, size=51)
    c = 51 // 2
    assert big[c, c + 5] / big[c, c] == pytest.approx(0.5, rel=1e-6)
    assert big[c + 5, c] / big[c, c] == pytest.approx(0.5, rel=1e-6)

    # larger fwhm spreads the kernel: the off-center/center value ratio grows
    narrow = util.get_gaussian_kernel(fwhm=2.0, size=15)
    wide = util.get_gaussian_kernel(fwhm=6.0, size=15)
    c = 7
    off = (c, c + 3)
    center = (c, c)
    assert wide[off] / wide[center] > narrow[off] / narrow[center]


def test_build_meshgrid():
    # scene_size=4, pixel_scale=1: ceil(4/1)=4 -> bumped to 5 (must be odd)
    X, Y = util.build_meshgrid(scene_size=4.0, pixel_scale=1.0)

    assert X.shape == (5, 5)
    assert Y.shape == (5, 5)

    # endpoints are exactly +/- scene_size/2
    assert X[0, 0] == pytest.approx(-2.0)
    assert X[0, -1] == pytest.approx(2.0)
    assert Y[0, 0] == pytest.approx(-2.0)
    assert Y[-1, 0] == pytest.approx(2.0)

    # odd num_pix -> exact zero at the center
    assert X[2, 2] == 0.0
    assert Y[2, 2] == 0.0

    # default 'xy' indexing: X varies along axis 1, Y varies along axis 0
    np.testing.assert_allclose(X[0], np.linspace(-2.0, 2.0, 5))
    np.testing.assert_allclose(Y[:, 0], np.linspace(-2.0, 2.0, 5))
    # ...so X is constant down a column and Y is constant across a row
    np.testing.assert_allclose(X[:, 3], np.full(5, X[0, 3]))
    np.testing.assert_allclose(Y[3, :], np.full(5, Y[3, 0]))


def test_rotate_array():
    # use uint8 so PIL accepts the array on every platform/version
    arr = np.array([[10, 20, 30],
                    [40, 50, 60],
                    [70, 80, 90]], dtype=np.uint8)

    # shape is preserved (PIL.Image.rotate without expand=True keeps original size)
    rotated = util.rotate_array(arr, angle=45)
    assert rotated.shape == arr.shape

    # rotation by 0 is the identity
    np.testing.assert_array_equal(util.rotate_array(arr, 0), arr)

    # rotation by 360 is the identity
    np.testing.assert_array_equal(util.rotate_array(arr, 360), arr)

    # a non-identity rotation actually changes an asymmetric pattern
    asym = np.zeros((5, 5), dtype=np.uint8)
    asym[0, 0] = 255
    rotated_asym = util.rotate_array(asym, angle=90, fillcolor=0)
    assert not np.array_equal(rotated_asym, asym)
    # PIL rotates counter-clockwise by 90 degrees: (0, 0) -> (4, 0)
    assert rotated_asym[4, 0] == 255


def test_polar_to_cartesian():
    # at the origin, any angle gives (0, 0)
    x, y = util.polar_to_cartesian(0.0, 1.234)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)

    # cardinal directions on the unit circle
    cases = [
        (1.0, 0.0,           1.0,  0.0),
        (1.0, np.pi / 2,     0.0,  1.0),
        (1.0, np.pi,        -1.0,  0.0),
        (1.0, 3 * np.pi / 2, 0.0, -1.0),
    ]
    for r, theta, expected_x, expected_y in cases:
        x, y = util.polar_to_cartesian(r, theta)
        assert x == pytest.approx(expected_x, abs=1e-12)
        assert y == pytest.approx(expected_y, abs=1e-12)

    # diagonal: r=sqrt(2), theta=pi/4 -> (1, 1)
    x, y = util.polar_to_cartesian(np.sqrt(2), np.pi / 4)
    assert x == pytest.approx(1.0, abs=1e-12)
    assert y == pytest.approx(1.0, abs=1e-12)

    # negative r flips the point through the origin
    x, y = util.polar_to_cartesian(-1.0, 0.0)
    assert x == pytest.approx(-1.0)
    assert y == pytest.approx(0.0, abs=1e-12)

    # vectorises over numpy arrays
    r = np.array([1.0, 2.0])
    theta = np.array([0.0, np.pi / 2])
    x, y = util.polar_to_cartesian(r, theta)
    np.testing.assert_allclose(x, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(y, [0.0, 2.0], atol=1e-12)


@pytest.mark.parametrize(("radius", "angle"), [
    (2.0, -np.pi / 3.0),
    ([1.0, 2.0], np.array([0.0, np.pi / 2.0])),
    (np.array([-2.0, 0.0, 3.0], dtype=np.float32), np.float32(0.25)),
    (np.array([[1.0], [-2.0]]), np.array([0.0, np.pi / 3.0, np.pi])),
    (np.empty((0, 1)), np.array([0.0, np.pi])),
])
def test_polar_to_cartesian_preserves_formula_and_broadcasting(radius, angle):
    expected = (radius * np.cos(angle), radius * np.sin(angle))

    actual = util.polar_to_cartesian(r=radius, theta=angle)

    for coordinate, reference in zip(actual, expected):
        np.testing.assert_array_equal(coordinate, reference)
        assert np.shape(coordinate) == np.shape(reference)


def test_polar_to_cartesian_delegates_to_tdpy(monkeypatch):
    import tdpy

    calls = []
    expected = (object(), object())

    def convert(radius, angle):
        calls.append((radius, angle))
        return expected

    monkeypatch.setattr(tdpy, 'retr_xposypos', convert)

    assert util.polar_to_cartesian(r=2.0, theta=0.25) is expected
    assert calls == [(2.0, 0.25)]


def test_percent_change():
    # docstring examples
    assert util.percent_change(50, 75) == pytest.approx(50.0)
    assert util.percent_change(100, 80) == pytest.approx(-20.0)

    # no change
    assert util.percent_change(42, 42) == pytest.approx(0.0)

    # the denominator is abs(old): going from -100 to -80 is a +20% change,
    # not a -20% change
    assert util.percent_change(-100, -80) == pytest.approx(20.0)

    # sign change
    assert util.percent_change(-100, 100) == pytest.approx(200.0)

    # 10x growth
    assert util.percent_change(1, 10) == pytest.approx(900.0)


def test_percent_difference():
    # 0% when the values are equal
    assert util.percent_difference(10, 10) == pytest.approx(0.0)
    assert util.percent_difference(7.5, 7.5) == pytest.approx(0.0)

    # standard percent-difference formula: |a-b| / mean(a,b) * 100
    # (10, 20) -> 10 / 15 * 100 = 200/3
    assert util.percent_difference(10, 20) == pytest.approx(200.0 / 3.0)
    # (10, 15) -> 5 / 12.5 * 100 = 40
    assert util.percent_difference(10, 15) == pytest.approx(40.0)
    # (1, 3) -> 2 / 2 * 100 = 100
    assert util.percent_difference(1, 3) == pytest.approx(100.0)

    # symmetric in its arguments
    assert util.percent_difference(7, 13) == pytest.approx(util.percent_difference(13, 7))


def test_center_crop_image():
    # cropping to the original shape returns the same object (identity)
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    same = util.center_crop_image(arr, (3, 3))
    assert same is arr

    # 5x5 -> 3x3 returns the centered 3x3 block
    arr = np.arange(25).reshape(5, 5)
    result = util.center_crop_image(arr, (3, 3))
    expected = np.array([[6, 7, 8],
                         [11, 12, 13],
                         [16, 17, 18]])
    np.testing.assert_array_equal(result, expected)

    # 7x7 -> 1x1 returns the single center pixel
    arr = np.arange(49).reshape(7, 7)
    result = util.center_crop_image(arr, (1, 1))
    np.testing.assert_array_equal(result, np.array([[24]]))

    # rectangular crop: 5x5 -> (3, 5) takes the center 3 rows, all columns
    arr = np.arange(25).reshape(5, 5)
    result = util.center_crop_image(arr, (3, 5))
    np.testing.assert_array_equal(result, arr[1:4, :])
    assert result.shape == (3, 5)


def test_center_crop_image_rejects_even_dimensions():
    odd = np.zeros((5, 5))
    even_src = np.zeros((4, 4))

    # even input array
    with pytest.raises(ValueError, match='Input array must have odd dimensions'):
        util.center_crop_image(even_src, (3, 3))

    # even target shape (one axis even is enough)
    with pytest.raises(ValueError, match='Requested shape must have odd dimensions'):
        util.center_crop_image(odd, (2, 3))
    with pytest.raises(ValueError, match='Requested shape must have odd dimensions'):
        util.center_crop_image(odd, (3, 2))

    # target larger than source
    with pytest.raises(ValueError, match='larger than input shape'):
        util.center_crop_image(odd, (7, 7))
