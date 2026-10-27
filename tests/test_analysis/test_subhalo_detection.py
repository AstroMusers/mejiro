import numpy as np

from mejiro.analysis.subhalo_detection import (
    detectable_fraction_by_mass,
    subhalo_parameters,
)


def test_subhalo_parameters_flattens_runs_in_input_order():
    detections = [
        [(0.2, 1.0e4, 5.0), (0.3, 1.0e5, 8.0)],
        [],
        [(0.4, 1.0e4, 6.0)],
    ]

    assert subhalo_parameters(detections) == (
        [0.2, 0.3, 0.4],
        [1.0e4, 1.0e5, 1.0e4],
        [5.0, 8.0, 6.0],
    )


def test_detectable_fraction_counts_exact_bins_over_all_runs():
    masses = np.array([1.0e4, 1.0e5, 1.0e4, 1.0e6])  # [M_sun]
    log_mass_grid = np.array([4.0, 5.0, 6.0])

    fractions = detectable_fraction_by_mass(log_mass_grid, masses, realization_count=4)

    np.testing.assert_array_equal(fractions, [0.5, 0.25, 0.25])


def test_detectable_fraction_keeps_zero_bins_and_empty_realizations():
    fractions = detectable_fraction_by_mass(
        np.array([4.0, 5.0, 6.0]), np.array([1.0e4]), realization_count=2
    )

    np.testing.assert_array_equal(fractions, [0.5, 0.0, 0.0])