import numpy as np
import pytest

from mejiro.analysis import snr_calculation


class _StubExposure:
    """Minimal duck-typed stand-in for `mejiro.exposure.Exposure`. The SNR
    calculation only reads the three array attributes."""
    def __init__(self, source_data, lens_data, data):
        self.source_data = source_data
        self.lens_data = lens_data
        self.data = data


def test_validate_raises_when_source_missing():
    exp = _StubExposure(source_data=None, lens_data=np.ones((3, 3)), data=np.ones((3, 3)))
    with pytest.raises(ValueError, match='pieces=True'):
        snr_calculation._validate_exposure_for_snr_calculation(exp)


def test_validate_raises_when_lens_missing():
    exp = _StubExposure(source_data=np.ones((3, 3)), lens_data=None, data=np.ones((3, 3)))
    with pytest.raises(ValueError, match='pieces=True'):
        snr_calculation._validate_exposure_for_snr_calculation(exp)


def test_validate_passes_when_both_present():
    exp = _StubExposure(
        source_data=np.ones((3, 3)),
        lens_data=np.ones((3, 3)),
        data=np.ones((3, 3)),
    )
    snr_calculation._validate_exposure_for_snr_calculation(exp)


def test_get_snr_array_basic():
    # source = 4 everywhere, total = 16 everywhere => SNR per pixel = 4 / 4 = 1
    source = np.full((3, 3), 4.0)
    data = np.full((3, 3), 16.0)
    exp = _StubExposure(source_data=source, lens_data=np.zeros((3, 3)), data=data)

    snr_arr = snr_calculation.get_snr_array(exp)
    np.testing.assert_allclose(snr_arr, np.ones((3, 3)))


def test_get_snr_array_per_pixel_values():
    source = np.array([[1.0, 2.0, 3.0]])
    data = np.array([[1.0, 4.0, 9.0]])
    exp = _StubExposure(source_data=source, lens_data=np.zeros((1, 3)), data=data)

    snr_arr = snr_calculation.get_snr_array(exp)
    np.testing.assert_allclose(snr_arr, np.array([[1.0, 1.0, 1.0]]))


def test_get_snr_array_replaces_nan_and_inf_with_zero():
    # data=0 -> 0/sqrt(0) = nan; source>0 with data=0 -> +inf; both should become 0
    source = np.array([[0.0, 5.0, 4.0]])
    data = np.array([[0.0, 0.0, 16.0]])
    exp = _StubExposure(source_data=source, lens_data=np.zeros((1, 3)), data=data)

    snr_arr = snr_calculation.get_snr_array(exp)
    np.testing.assert_allclose(snr_arr, np.array([[0.0, 0.0, 1.0]]))
    assert np.all(np.isfinite(snr_arr))


def test_get_snr_array_raises_without_pieces():
    exp = _StubExposure(source_data=None, lens_data=None, data=np.ones((3, 3)))
    with pytest.raises(ValueError, match='pieces=True'):
        snr_calculation.get_snr_array(exp)


def test_get_snr_returns_one_when_all_below_threshold():
    # SNR per pixel = 1 / sqrt(100) = 0.1, well below default threshold of 1
    source = np.ones((4, 4))
    data = np.full((4, 4), 100.0)
    exp = _StubExposure(source_data=source, lens_data=np.zeros((4, 4)), data=data)

    max_snr, masked = snr_calculation.get_snr(exp)
    assert max_snr == 1
    assert masked.mask.all()


def test_get_snr_threshold_boundary_is_inclusive():
    # snr = 1 exactly should be MASKED (the implementation uses `<= threshold`)
    source = np.zeros((3, 3))
    data = np.full((3, 3), 1.0)
    source[1, 1] = 2.0
    data[1, 1] = 4.0   # snr = 2/2 = 1.0 == threshold

    exp = _StubExposure(source_data=source, lens_data=np.zeros((3, 3)), data=data)
    max_snr, masked = snr_calculation.get_snr(exp, snr_per_pixel_threshold=1)

    assert max_snr == 1
    assert masked.mask.all()


def test_get_snr_single_region_single_pixel():
    # one pixel above threshold; SNR_region = source / sqrt(total)
    source = np.zeros((5, 5))
    data = np.ones((5, 5))   # background pixels: snr = 0/1 = 0, all masked
    source[2, 2] = 10.0
    data[2, 2] = 25.0        # snr_per_pixel = 10/5 = 2 > 1

    exp = _StubExposure(source_data=source, lens_data=np.zeros((5, 5)), data=data)
    max_snr, masked = snr_calculation.get_snr(exp)

    assert max_snr == pytest.approx(10.0 / np.sqrt(25.0))   # = 2.0
    assert masked.mask.sum() == 24
    assert not masked.mask[2, 2]


def test_get_snr_single_region_multiple_pixels():
    # two horizontally adjacent pixels => one region under cross connectivity
    source = np.zeros((5, 5))
    data = np.ones((5, 5))
    source[2, 2] = 10.0
    source[2, 3] = 10.0
    data[2, 2] = 25.0
    data[2, 3] = 25.0

    exp = _StubExposure(source_data=source, lens_data=np.zeros((5, 5)), data=data)
    max_snr, _ = snr_calculation.get_snr(exp)

    # SNR_region = (10 + 10) / sqrt(25 + 25) = 20 / sqrt(50) = 2*sqrt(2)
    assert max_snr == pytest.approx(2 * np.sqrt(2))


def test_get_snr_disjoint_regions_returns_max():
    # two well-separated single-pixel regions; the brighter one wins
    source = np.zeros((7, 7))
    data = np.ones((7, 7))
    source[1, 1] = 10.0
    data[1, 1] = 25.0   # SNR = 2
    source[5, 5] = 20.0
    data[5, 5] = 25.0   # SNR = 4

    exp = _StubExposure(source_data=source, lens_data=np.zeros((7, 7)), data=data)
    max_snr, _ = snr_calculation.get_snr(exp)

    assert max_snr == pytest.approx(20.0 / np.sqrt(25.0))   # = 4.0


def test_get_snr_diagonal_pixels_are_not_merged():
    # cross-shaped (4-connectivity) connectivity: diagonal neighbors are SEPARATE regions
    source = np.zeros((4, 4))
    data = np.ones((4, 4))
    source[1, 1] = 10.0
    data[1, 1] = 25.0
    source[2, 2] = 10.0
    data[2, 2] = 25.0

    exp = _StubExposure(source_data=source, lens_data=np.zeros((4, 4)), data=data)
    max_snr, _ = snr_calculation.get_snr(exp)

    # if merged into one region, SNR would be 20/sqrt(50) = 2*sqrt(2) ~ 2.828
    # under 4-connectivity, two separate regions each with SNR = 10/5 = 2
    assert max_snr == pytest.approx(2.0)
    assert max_snr != pytest.approx(2 * np.sqrt(2))


def test_get_snr_threshold_parameter_changes_regions():
    # at threshold=1, both bright pixels are above the cut and form their own
    # disjoint single-pixel regions; the dim pixel between them is masked.
    # at threshold=0.5, the dim pixel (snr~0.6) joins them, fusing the two
    # single-pixel regions into one larger region with a different SNR.
    source = np.zeros((1, 5))
    data = np.ones((1, 5))
    source[0, 1] = 10.0
    data[0, 1] = 25.0          # snr = 2
    source[0, 2] = 3.0
    data[0, 2] = 25.0          # snr = 0.6
    source[0, 3] = 10.0
    data[0, 3] = 25.0          # snr = 2

    exp = _StubExposure(source_data=source, lens_data=np.zeros((1, 5)), data=data)

    high_thresh, _ = snr_calculation.get_snr(exp, snr_per_pixel_threshold=1)
    low_thresh, _ = snr_calculation.get_snr(exp, snr_per_pixel_threshold=0.5)

    # high threshold: two disjoint single-pixel regions, max SNR = 2
    assert high_thresh == pytest.approx(2.0)
    # low threshold: one fused region, SNR = (10+3+10) / sqrt(25+25+25) = 23 / sqrt(75)
    assert low_thresh == pytest.approx(23.0 / np.sqrt(75.0))


def test_get_snr_uses_total_data_not_source_in_denominator():
    # SNR_region uses sum of `data` (= source + lens + sky + noise) in the
    # denominator, not just source. Lens flux in the same pixel must dilute
    # the SNR.
    source_only = np.zeros((3, 3))
    source_only[1, 1] = 10.0

    data_source_alone = np.ones((3, 3))
    data_source_alone[1, 1] = 25.0   # data == source contribution; SNR_pixel = 2

    data_with_lens = np.ones((3, 3))
    data_with_lens[1, 1] = 100.0     # source + lens + ...; SNR_pixel = 10/10 = 1
    # boost it just over threshold so the pixel survives the mask
    data_with_lens[1, 1] = 99.0       # SNR_pixel = 10/sqrt(99) ~ 1.005 > 1

    exp_no_lens = _StubExposure(
        source_data=source_only, lens_data=np.zeros((3, 3)), data=data_source_alone
    )
    exp_with_lens = _StubExposure(
        source_data=source_only, lens_data=np.zeros((3, 3)), data=data_with_lens
    )

    snr_no_lens, _ = snr_calculation.get_snr(exp_no_lens)
    snr_with_lens, _ = snr_calculation.get_snr(exp_with_lens)

    assert snr_no_lens == pytest.approx(10.0 / np.sqrt(25.0))
    assert snr_with_lens == pytest.approx(10.0 / np.sqrt(99.0))
    assert snr_with_lens < snr_no_lens
