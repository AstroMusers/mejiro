"""Quality checks on the exposures in an exported HDF5 dataset.

The check here exists because of the defect in docs/l3_negative_drizzle_weights.md: romancal's
default ``ivm`` resample weighting derived negative drizzle weights on bright deflector cores,
and the resulting ``sum(w*d)/sum(w)`` produced isolated spikes, near-zeros and negative
surface brightness in 246 of the 322,521 exposures of roman_data_challenge_rung_1 v3.0. None
of it was visible in aggregate statistics, so it shipped.

Corruption of that kind is local by construction -- one pixel disagreeing violently with its
own neighbours -- so it is caught by comparing every pixel to the median of its eight
neighbours. The thresholds below come from the measured baseline of that dataset: over the
290,268 exposures faint enough that they cannot be affected (the bottom 90% by peak surface
brightness, hence necessarily clean), the neighbour ratio spans [0.79, 3.30]. Flagging
outside [0.5, 5.0] therefore leaves ~1.5x headroom on both sides while still catching every
one of the 199 corrupted systems.

The check is only meaningful on an L3 co-add, whose background is smoothed by drizzling
several exposures together. A single-exposure galsim frame is noise-dominated pixel to pixel
and would trip these thresholds routinely, which is why _06_h5_export only runs this for
romanisim input.
"""
import h5py
import numpy as np

import logging

logger = logging.getLogger(__name__)

MAX_NEIGHBOR_RATIO = 5.0
MIN_NEIGHBOR_RATIO = 0.5

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def neighbor_ratio(data):
    """Ratio of each interior pixel to the median of its eight neighbours.

    Returns an ``(n-2, m-2)`` array; the one-pixel border has no complete neighbourhood and
    is not checked.
    """
    n, m = data.shape
    interior = data[1:-1, 1:-1]
    ring = np.median(
        np.stack([data[1 + dy:n - 1 + dy, 1 + dx:m - 1 + dx] for dy, dx in _NEIGHBORS]),
        axis=0,
    )
    return interior / ring


def find_corrupted_exposures(filepath, max_ratio=MAX_NEIGHBOR_RATIO, min_ratio=MIN_NEIGHBOR_RATIO):
    """Find exposures containing non-physical pixels in an exported HDF5 dataset.

    Flags an exposure if it contains a non-finite pixel, a negative pixel (the co-add is a
    positive-definite surface brightness, so a negative value can only come from a division
    by a vanishing drizzle weight), or a pixel outside
    ``[min_ratio, max_ratio]`` times the median of its eight neighbours.

    Returns a list of ``(uid, band, reason)`` tuples, one per offending exposure, where
    ``reason`` names the worst pixel and its value.
    """
    findings = []
    with h5py.File(filepath, 'r') as f:
        for group_name, group in f['images'].items():
            uid = group_name.removeprefix('strong_lens_')
            for dset_name in group:
                if not dset_name.startswith('exposure_'):
                    continue
                band = dset_name.split('_')[-1]
                data = group[dset_name][()].astype(np.float64)

                if not np.isfinite(data).all():
                    findings.append((uid, band, f'{(~np.isfinite(data)).sum()} non-finite pixel(s)'))
                    continue

                if data.min() < 0:
                    y, x = np.unravel_index(data.argmin(), data.shape)
                    findings.append((uid, band, f'negative pixel {data[y, x]:.4g} at ({y}, {x})'))
                    continue

                ratio = neighbor_ratio(data)
                if ratio.max() > max_ratio:
                    y, x = np.unravel_index(ratio.argmax(), ratio.shape)
                    findings.append((uid, band, f'pixel at ({y + 1}, {x + 1}) is {ratio.max():.4g}x '
                                                f'its neighbour median ({data[y + 1, x + 1]:.4g})'))
                elif ratio.min() < min_ratio:
                    y, x = np.unravel_index(ratio.argmin(), ratio.shape)
                    findings.append((uid, band, f'pixel at ({y + 1}, {x + 1}) is {ratio.min():.4g}x '
                                                f'its neighbour median ({data[y + 1, x + 1]:.4g})'))
    return findings


def check_exposures(filepath, max_ratio=MAX_NEIGHBOR_RATIO, min_ratio=MIN_NEIGHBOR_RATIO):
    """Raise if any exposure in ``filepath`` contains non-physical pixels.

    Deliberately raises rather than repairing: a corrupted pixel means the co-add that
    produced it is wrong, and the fix belongs upstream in _05_romanisim, not here.
    """
    findings = find_corrupted_exposures(filepath, max_ratio=max_ratio, min_ratio=min_ratio)
    if not findings:
        logger.info(f'Exposure check passed: no corrupted pixels in {filepath}')
        return

    systems = len({uid for uid, _, _ in findings})
    for uid, band, reason in findings[:20]:
        logger.error(f'  {uid} {band}: {reason}')
    if len(findings) > 20:
        logger.error(f'  ... and {len(findings) - 20} more')
    raise ValueError(
        f'{len(findings)} corrupted exposure(s) across {systems} system(s) in {filepath}; '
        f'see docs/l3_negative_drizzle_weights.md'
    )
