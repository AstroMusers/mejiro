# Oversampled step-04 rendering: removing a double pixel-response convolution

**Status:** diagnosed and fixed 2026-07-26. Every dataset produced before this change is
affected, at the few-percent-of-peak level. The fix is the `synthetic_image.oversample`
config knob; `oversample: 1` reproduces the old behaviour exactly, so only configs that
opt in change.

This is the first of three changes landed together against the step-04/05 resampling path.
The other two are [l3_dither_registration.md](l3_dither_registration.md) (which this one is
a prerequisite for) and [subpixel_phase.md](subpixel_phase.md).

## TL;DR

Step 04 rendered at the detector pixel scale, which meant lenstronomy applied the detector
pixel response **twice**: once by binning the supersampled surface brightness to 0.11"/px,
and again through the PSF kernel, which lenstronomy box-averages to that same scale. The
effective PSF of every mejiro image was therefore broader than the STPSF PSF that
`psf.oversamples` and the `cached_psfs` directory hand to a modeler.

The fix renders at `pixel_scale / oversample` and convolves there with the oversampled STPSF
kernel used as-is. The single detector-pixel integral then happens exactly once, in step 05,
where it belongs — and where it can be done at the correct sub-pixel phase.

## Root cause

`_04_create_synthetic_images.py` passed
`kwargs_numerics = {'supersampling_factor': 5, 'compute_mode': 'adaptive'}` and never set
`supersampling_convolution`, which lenstronomy defaults to `False`. In that configuration
`lenstronomy/ImSim/Numerics/numerics.py` takes the `else` branch and convolves with
`psf.kernel_point_source`.

mejiro supplies the oversampled STPSF kernel with `point_source_supersampling_factor=5`
(`mejiro/utils/lenstronomy_util.py::get_pixel_psf_kwargs`), so lenstronomy derives
`kernel_point_source` from it via `kernel_util.degrade_kernel` — an exact 5x5 box average.
That box average *is* the detector pixel response.

So the model was:

```
re_size(S_5x)  (x)  degrade(K_5x)   ==   (S (x) P) (x) (K (x) P)
```

where `P` is the 0.11" pixel top-hat, when the detector actually records `(S (x) K) (x) P`.
One extra `P`.

Verified two ways:

- **Analytically**, on the kernel alone: the second moment of `degrade_kernel(K, 5)` equals
  `sigma^2(K)/25 + (5^2-1)/12/5^2` to 5 decimal places, i.e. exactly one box variance added.
  Pinned by `tests/test_utils/test_lenstronomy_util.py::test_degrading_adds_exactly_one_pixel_response`.
- **End to end**, rendering a Sersic n=4, Re=0.35" through lenstronomy both ways: the old
  path loses **3.6% of the peak**, differs by up to 4.2% of peak per-pixel, and has ~1.1%
  median fractional error over pixels above 1% of peak.

Note the double `P` cancels for a source sitting exactly at a pixel center — there
`S (x) P` is still a delta, and `degrade_kernel(K)` and a phase-0 `bin_to_native(K)` are the
same block sum. A point-source test therefore shows no difference at all; the error is real
only for extended structure, which is everything the pipeline actually renders. (That
degeneracy is not a coincidence: every system *was* sitting at a pixel center — see
[subpixel_phase.md](subpixel_phase.md).)

On three real substructured rung-1 lenses (F129, SCA 01, detector position pinned), the old
and new renders differ by **2.9-4.2% of peak** and 0.24-0.69% median per-pixel, with the new
peak 0.6-3.5% lower. Total flux agrees to 0.02-0.24%. Be careful attributing that sign to
the pixel-response term alone: the new config also ray-shoots the surface brightness on a
much finer grid (0.0044" vs 0.022"), which integrates a cuspy deflector far more accurately,
and for these systems that dominates. The pixel-response claim rests on the kernel-level and
Sersic measurements above, which isolate it.

## The fix

`SyntheticImage(oversample=N)` builds the pixel grid at `pixel_scale / N` with
`num_pix * N` pixels. The native grid is sized first and the oversampled one derived from
it, because sizing directly at the fine scale does not give a multiple:
`set_odd_num_pix(10.01, 0.022)` is 457, not `5 * set_odd_num_pix(10.01, 0.11)` = 455.
`oversample` must be odd so the oversampled grid stays centered on the native one.

The kernel must then be used at that resolution rather than degraded, which is what the new
`get_pixel_psf_kwargs(..., degrade=False)` selects. `SyntheticImage` raises if
`oversample > 1` is combined with a kernel lenstronomy would degrade, so the two cannot
drift apart.

`oversample` rides in the lightweight `.npz` metadata (`schema_version` 2) alongside
`pixel_scale` and `num_pix`, so step 05 derives the binning factor from the file itself
rather than from a sidecar that could go stale. Version 1 files still load and report
`oversample = 1`; they describe valid detector-resolution data.

## Choosing `supersampling_factor`

At `oversample > 1`, `supersampling_factor` subdivides an already-fine subpixel rather than
a detector pixel, so its meaning — and its right value — changes. Measured per lens-band on
three real substructured rung-1 systems (non-JAX path, F129, SCA 01), converged against
`supersampling_factor: 5`, all compared after binning to detector pixels:

| config | s/lens-band | median per-pixel vs ssf=5 | max, as a fraction of peak |
|---|---|---|---|
| `oversample: 1`, `supersampling_factor: 5` (the old setting) | 5.70 | n/a, different PSF treatment | |
| `oversample: 5`, `supersampling_factor: 1` | 3.97 | 0.017-0.110% | 0.15-0.70% |
| `oversample: 5`, `supersampling_factor: 3` | 29.88 | 0.001-0.002% | 0.04-0.14% |
| `oversample: 5`, `supersampling_factor: 5` | 71.6 | reference | |

The rung-1 configs use **`supersampling_factor: 3`**: 0.70% of peak is not comfortably below
the substructure signal this dataset exists to measure, and 0.14% is. That is ~5x the old
config's cost, on a step that takes ~20 min under JAX.

A smooth analytic test lens (`Sample1`) initially suggested `supersampling_factor: 1` was
accurate to 0.035% median. It is not representative — real systems carry pyHalo substructure
and a cuspy deflector with structure below 0.022" — so this table uses real inputs.

Note `oversample: 5` is not inherently a cost blow-up: at `supersampling_factor: 1` it is
*cheaper* than the old config (3.97 s vs 5.70 s), because the old adaptive path was already
ray-shooting at 0.022" over the annulus. The cost here is bought deliberately, by the
`supersampling_factor: 3` choice.

## Scope

- `_05_galsim.py` raises on `oversample > 1`: the galsim engine assumes detector-resolution
  input. `ella` and any other galsim-path dataset therefore keep `oversample: 1` and are
  unaffected by this change.
- `_06_h5_export.py` labels exposure datasets with `synthetic_image.native_pixel_scale`
  rather than `pixel_scale`; the exposures are always on the detector grid even when the
  step-04 image is not.
- `calculate_snrs.py` reads step 04 only for `instrument_params` and builds its own
  `SyntheticImage`, so it is unaffected.
- Step-04 output grows 25x at `oversample: 5` (13 GB -> ~270 GB for rung-1).

## `detector_position` was drawn with an unseeded `random.choice` (fixed)

`_04_create_synthetic_images.create_synthetic_image` used to do:

```python
detector_position = random.choice(possible_detector_positions)
```

Nothing seeded the global `random` module, so the detector position — and therefore which
STPSF kernel a system was convolved with — differed every time step 04 ran. It was stable
across bands (chosen once per lens, before the band loop) and recorded in the `.psfpos.json`
sidecar, so a finished dataset was self-documenting. But it was not reproducible: re-running
step 04 with the same `seed`, or filling in systems with `--resume`, assigned different PSFs
than the original run did.

This surfaced while measuring the effect of this change — the first pass compared step-04
renders across runs and was actually comparing different PSFs. Every number in this document
comes from runs with the position pinned.

Replaced by `_detector_position(name, seed, possible_positions)`, hashed on `seed` + lens
name exactly as `_is_deflector_only` beside it already was, and indexing the position list by
`int(h[:8], 16) % len(possible_positions)`. Verified end to end: two renders of the same three
lenses now assign identical positions, where before they differed on every run. The modulo is
uniform to within the hash's own uniformity, which matters because `_05_romanisim`'s L2 path
buckets systems by detector position and round-robins batches across those buckets;
`tests/test_pipeline/test_detector_position.py` asserts that over 16000 names.

Note `_is_deflector_only` keys on the input *filename* while this keys on `StrongLens.name`
(what names the outputs, and what the tail steps parse uids from). Both are stable; they
simply draw from independent streams.

**This changes which PSF every Roman system gets, so it changes every pixel**, and needs the
same rebuild as the rest of this work.

## Adjacent, not fixed

**`SyntheticImage.__init__` mutates the caller's `kwargs_numerics` dict**, caching
`supersampled_indexes` into it. Reusing one dict across constructions of different sizes
silently reuses the wrong adaptive mask (it surfaces as a numpy broadcast error). Harmless
in the pipeline, where the reuse is across bands at identical geometry, but a trap for
callers and tests.
