"""Utilities for summarizing simulated subhalo detection results."""

import numpy as np


def subhalo_parameters(detections):
    """Flatten per-realization ``(redshift, mass, concentration)`` records."""
    redshifts = []
    masses = []
    concentrations = []
    for realization in detections:
        for redshift, mass, concentration in realization:
            redshifts.append(redshift)
            masses.append(mass)
            concentrations.append(concentration)
    return redshifts, masses, concentrations


def detectable_fraction_by_mass(log_mass_grid, masses, realization_count):
    """Count exact log-mass matches and divide by all realizations."""
    log_masses = np.log10(np.asarray(masses))
    return [
        np.sum(log_masses == log_mass) / realization_count
        for log_mass in log_mass_grid
    ]