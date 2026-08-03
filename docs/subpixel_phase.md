# Sub-pixel phase: every system was centered on a detector pixel

**Status:** diagnosed and fixed 2026-07-26. Affects every dataset produced before this
change, at both L2 and L3. The fix requires `synthetic_image.oversample > 1`; at
`oversample: 1` there is no valid way to apply a phase and the behaviour is unchanged.

Third of three changes landed together against the step-04/05 resampling path — see
[step04_oversampled_rendering.md](step04_oversampled_rendering.md) and
[l3_dither_registration.md](l3_dither_registration.md).

## TL;DR

The step-04 pixel grid is centered on the deflector, and step 05 placed every tile with its
center on an integer detector pixel. So every system in every mejiro dataset sat at
sub-pixel phase (0, 0) — its centroid dead-center on a pixel. That is the single most
favorable sampling phase there is for an undersampled PSF, and a real survey delivers a
uniform distribution of phases instead.

Measured on 400 real `roman_data_challenge_rung_1` step-04 files (SCA 01, F129):
**97.5% have their brightest pixel at the exact center index (45, 45)**; the median offset
on both axes is 0.

## Root cause

Two independent pieces line up:

1. `SyntheticImage` builds its grid with
   `lenstronomy_util.make_grid_with_coordtransform(..., left_lower=False)`, which centers
   the grid on (0, 0) in angular coordinates — where the deflector sits.
2. `_place_tile` and the L2 tiling both position the tile at integer detector pixels
   (`counts[r0:r0+tile_size, ...]` with `r0 = tile_r * tile_size`, and
   `int(round(src_pix.x))` for L3).

`tile_size` is odd, so the tile's center pixel lands on a whole detector pixel and the
deflector centroid lands at its center.

## Why it matters

Roman WFI is undersampled at 0.11"/px. How much of the PSF core falls in the peak pixel
depends strongly on where the source sits within that pixel — that dependence is exactly
what a sub-pixel dither pattern exists to exploit. Fixing the phase at 0 for every system:

- biases the peak pixel high and the immediate neighbours low, by an amount that does not
  average out over the population because it is the same for every system;
- removes a source of realistic variance from the dataset, so anything trained or
  calibrated on it sees a cleaner core than real data provides;
- is the most favorable case, so any performance measured on it is optimistic.

## The fix

L3 has a real phase available and now uses it: each dither's true position carries a
fractional residual, which `bin_to_native` applies before binning
(see [l3_dither_registration.md](l3_dither_registration.md)).

L2 has no dither WCS to derive one from, so `process_batch_l2` draws a deterministic phase
per system:

```python
def subpixel_phase(seed, lens_name):
    h = hashlib.md5(f'{seed}_subpixel_phase_{lens_name}'.encode()).hexdigest()
    return (int(h[:8], 16) / 0x100000000 - 0.5,
            int(h[8:16], 16) / 0x100000000 - 0.5)
```

Keyed on `seed` + lens name rather than list order, mirroring
`_04_create_synthetic_images._is_deflector_only`, so the phase is identical in every band
and stable across `--resume` and the reordering both steps do. Uniform on `[-0.5, 0.5)` per
axis; `tests/test_pipeline/test_subpixel_phase.py` pins determinism, range, uniformity and
order-independence.

## Scope

Only datasets built through `_05_romanisim.py` with `oversample > 1`. `ella` and other
galsim-path datasets go through `_05_galsim.py`, which rejects oversampled input, so they
keep the old pixel-centered behaviour.
