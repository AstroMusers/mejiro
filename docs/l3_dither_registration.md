# L3 dither registration: integer tile placement misregisters the dithers

**Status:** diagnosed 2026-07-26; **fixed 2026-07-26** in `process_batch_l3` via
`bin_to_native`, which requires `synthetic_image.oversample > 1` from step 04 (see
[step04_oversampled_rendering.md](step04_oversampled_rendering.md)). Every L3 dataset
produced before the fix is affected, including `roman_data_challenge_rung_1` v3.0, and
needs a step-04 + step-05 rebuild. Found while investigating a separate defect (corrupted
bright-core pixels from the resample `ivm` weighting); deliberately kept out of that fix so
the two changes stay separately attributable.

**Resolution:** the fix is in the "Fix" section below, which supersedes the original
"Proposed fix". The measurements that drove the choice are there too: the option originally
preferred (shifting the native-resolution tile) turned out to be ~10% wrong on the PSF core,
because a 0.11"/px tile is undersampled and cannot be interpolated correctly.

## TL;DR

`process_batch_l3` renders each system's tile at an **integer** detector pixel in every
dither, but the L3 co-add combines the dithers using their **true** WCS. Dither 0 is exact by
construction; dithers 1..N-1 each carry an independent rounding residual of up to ±0.5 px in
each axis. The dithers are therefore mutually misregistered by up to ~0.85 px, so the co-add
is an average of N sub-pixel-shifted copies of the same system rather than N registered ones.

This is worst for exactly the patterns it matters most for. `roman_data_challenge_rung_1`
v3.0 was produced with **SUB6**, whose commanded offsets are 0.019-0.090 arcsec — that is
**0.17 to 0.82 detector pixels**. A sub-pixel pattern exists to sample the PSF at controlled
sub-pixel phases; `int(round(...))` snaps every one of those phases onto a whole pixel, so
the commanded pattern is quantized away and replaced by an uncontrolled rounding residual.
The dithering is not merely degraded, it is doing something other than what was asked for.

Measured on the rung-1 SUB6 configuration: mutual misregistration averages 0.70 px (max
0.85 px), **~2.8% of the core peak is lost** (worst case 5.6%), and the effective co-added
PSF is broadened by a per-system-random amount. This is a systematic, deterministic bias —
not noise.

## Symptom

There is no dramatic visual artifact. The consequences are quantitative:

- Core peak surface brightness is systematically low by a few percent.
- **The effective PSF of the co-add is broader than the one the images were built with.** The
  step-04 `SyntheticImage`s are convolved with the STPSF PSF for the system's detector
  position, and step 05 then co-adds N copies of that displaced by the rounding residuals —
  an extra convolution nobody asked for. Any lens modeling that assumes the STPSF PSF (the
  natural choice, and what `psf.oversamples` / the `cached_psfs` directory provide) is using
  a PSF that is too narrow. Note the current `_06_h5_export.py` does not write a `psfs` group
  at all and every project config sets `include_psfs: False`, so this is a modeling-side
  mismatch rather than a self-inconsistency inside the h5.
- The amount of broadening varies per system (it depends on where the tile sits in the
  overlap grid), so it cannot be absorbed into a single corrected PSF.

## Root cause

### 1. The tile grid is built at integer dither-0 pixel positions

`compute_overlap_skygrid` ([_05_romanisim.py:259](../mejiro/pipeline/_05_romanisim.py#L259))
lays out tile centers on an integer pixel grid in dither-0 coordinates and converts them to
sky:

```python
source_skies.append(wcses[0].toWorld(galsim.PositionD(gx, gy)))
```

`gx`, `gy` are integers, so each tile's nominal sky position is exactly on a dither-0 pixel
center. This is why dither 0 alone comes out exact.

### 2. Each dither re-renders the tile at a rounded pixel position

In `process_batch_l3` ([_05_romanisim.py:803-805](../mejiro/pipeline/_05_romanisim.py#L803-L805)):

```python
for e, sky in zip(electrons, source_skies):
    src_pix = wcses[d].toImage(sky)
    _place_tile(counts, e, int(round(src_pix.x)), int(round(src_pix.y)), tile_size)
```

`_place_tile` ([_05_romanisim.py:309](../mejiro/pipeline/_05_romanisim.py#L309)) does an
array slice assignment — it can only place a tile on whole pixels. The `int(round(...))`
therefore displaces the rendered system from its true sky position by the rounding residual,
independently in each dither.

### 3. The co-add registers on sky, not on the rounded positions

`MosaicPipeline`'s resample step drizzles each L2 through its own WCS onto the mosaic grid.
It has no knowledge of the rounding, so it places dither *d*'s rendering at the position the
WCS says it belongs — which is offset from where the flux actually was rendered. The N
renderings land at N different mosaic positions and are averaged there.

`_extract_cutout` then slices around the *true* sky position via `mos_wcs.world_to_pixel`, so
the cutout is correctly centered on average; the damage is the intra-stack spread, not a net
centroid shift.

## Evidence

Measured with the pipeline's own functions at the rung-1 pointing (RA 150°, Dec +2°, MA table
10, 2027-05-01, `tile_size = 91`), over ~60 tile slots per SCA. Both the pattern rung-1 v3.0
actually used (SUB6, capacity 1600 slots/batch, matching the production `batch_complete_*.txt`
counts) and a gap-filling pattern for comparison:

| pattern | SCA | mean \|rounding residual\| | max | mean mutual misregistration | max |
|---------|-----|---------------------------|-----|----------------------------|-----|
| SUB6      | 01 | 0.215 px | 0.498 px | 0.701 px | 0.830 px |
| SUB6      | 09 | 0.209 px | 0.498 px | 0.684 px | 0.849 px |
| BOXGAP6_1 | 01 | 0.244 px | 0.499 px | 0.831 px | 0.890 px |
| BOXGAP6_1 | 09 | 0.232 px | 0.490 px | 0.802 px | 0.910 px |

"Mutual misregistration" is `max - min` of the per-dither residual across the 6 dithers, per
axis — i.e. how far apart the extreme dithers place the same system. **No tile slot had all
six dithers round to the same pixel** (0 of 62 sampled, both SCAs), so no system escapes it.

Per-dither SUB6 residuals for one representative tile slot (dy, dx in detector px):

```
dither 0: (+0.000, -0.000)   <- exact by construction (grid is integer in dither-0 pixels)
dither 1: (-0.164, -0.167)
dither 2: (+0.171, +0.180)
dither 3: (+0.408, -0.332)
dither 4: (+0.247, +0.498)
dither 5: (+0.085, +0.330)
```

Averaging cubic-shifted copies of real step-04 `SyntheticImage`s at those offsets (20 tile
slots x 3 systems) and comparing to the correctly-registered average:

```
co-add peak retained:  mean 0.972,  worst 0.944   (1.000 = perfectly registered)
```

Reproduce with `mejiro.pipeline._05_romanisim._dither_pointings` +
`compute_overlap_skygrid`, then `wcses[d].toImage(sky)` and take fractional parts.

## Fix

The tile is rendered at its true sub-pixel position by shifting it on an **oversampled**
grid and binning to detector pixels there, rather than by shifting a native-resolution tile.
`process_batch_l3` now does, per system per dither:

```python
src_pix = wcses[d].toImage(sky)
ix, iy = int(round(src_pix.x)), int(round(src_pix.y))
native = bin_to_native(smooth, ov, src_pix.x - ix, src_pix.y - iy)
_place_tile(counts, native, ix, iy, tile_size)
```

`DISTORTION_GUARD = 7` and the overlap-grid logic are unaffected — this changes only where
within its cell a tile is rendered, not the cell layout.

### Why the shift has to be on the oversampled grid

The option originally preferred here — shifting the 91x91 tile in place with
`scipy.ndimage.shift` — is wrong at the percent level, and for a reason that is easy to miss.
Roman WFI at 0.11"/px is undersampled (lambda/D ~ 0.129" at 1.5 um needs ~0.064"/px for
Nyquist), so the shift theorem does not apply to the tile: the interpolation is
aliasing-limited exactly at the PSF core, where the substructure signal is measured.

Measured against an exact reference (the real F129 STPSF kernel displaced on the oversampled
grid and then binned), over the six SUB6 residuals, median per-pixel error on pixels above
1% of peak:

| scheme | 1 exposure, worst offset | 6-dither co-add |
|---|---|---|
| integer placement (the defect) | 54% | 22.5% |
| shift the native tile, cubic spline | 48% | 9.9% |
| shift the native tile, Fourier | 130% | 16.6% |
| oversample 5x, bin at nearest 1/5 phase | 10.5% | 3.9% |
| **oversample 5x, shift there, then bin** | **exact** | **exact** |

Two things fall out of that table:

- Fourier shifting the native tile is *worse* than cubic, because it rings on the aliased
  power rather than smoothing over it.
- Oversampling without a shift is not enough either. At 5x the residual phase error is up to
  0.1 px, still ~4% on the co-add of a point source; reaching 0.1% by quantization alone
  would need oversample ~50. The oversampled render and the sub-pixel shift are both
  required.

On a resolved feature (Sersic n=1, Re=0.20") the native shift does fine — 0.04% on the
co-add — which is why this is easy to get wrong: it only fails on the unresolved core.

### Implementation notes

`bin_to_native` shifts with a Fourier phase ramp rather than `scipy.ndimage.shift`. Measured
on a realistic 455x455 tile: exactly flux-conserving (the DC term is untouched) versus ~1e-4
for splines, ~2-4x faster (15 ms vs 30-62 ms), and it introduces no negative pixels. Its
periodic wrap is harmless because tile edges sit at sky level, ~5e-5 of peak. That the
Fourier shift is valid on this grid is checked directly: an integer-subpixel shift reproduces
an exact `np.roll` to 2e-16 of peak.

Only a translation is applied. The per-dither local Jacobians were measured at three tile
slots on SCA 01 for the rung-1 pointing: the worst corner displacement of a 91-px tile from
Jacobian mismatch is **0.002 native px** (BOXGAP4_1) and 0.000 (SUB6), so an affine resample
would buy nothing.

Cost: one shift + block sum per system per dither, ~15 ms on 455x455. About 15 min added
across the whole ~7 h rung-1 L3 run at 36 workers.

## Validation

Covered by unit tests:

- `tests/test_pipeline/test_subpixel_binning.py` — zero phase is a plain block sum
  bit-for-bit; flux conserved to 1e-12 at arbitrary phases; a `k/oversample` shift reproduces
  an exact `np.roll`; direction and axis order match `_place_tile` (guarding the sign-flip
  class of bug behind [l3_cutout_orientation.md](l3_cutout_orientation.md)); and the accuracy
  comparison above, asserted against the real F129 kernel checked in at
  `tests/test_data/F129_1_2048_2048_5_101.npy`.
- `tests/test_pipeline/test_dither_subpixel_phases.py` — using the real SUB6 dither WCSes:
  dither 0 is exact, the other dithers carry residuals spanning >0.5 px with no tile slot
  escaping, and binning at those phases yields six *distinct* detector samplings rather than
  six copies.
- `tests/test_pipeline/test_mejiro_pipeline.py` still passes (`test.yaml`, `data_dir: null`,
  ~96 s). Note it runs the **galsim** engine, so it does not exercise this path; the romanisim
  path is covered by the tests above. Do not validate by running a real project config —
  `_05_romanisim.py` deletes all output in `output_dir` without `--resume`.

Still to do on the rebuilt data:

- Confirm the commanded sub-pixel pattern survives all the way to the mosaic: with SUB6 the
  six renderings should land at the six commanded phases.
- Compare a bright system's co-added cutout against `smooth_pixels(synth.data)` of its step-04
  input, as was done for the parity flip in
  [l3_cutout_orientation.md](l3_cutout_orientation.md) — the core should sharpen.
- Size the residual error against the substructure signal: for one lens with and without a
  pyHalo realization, compare the substructure-induced pixel differences to the residual
  error of the new path.

## Impact / regeneration

This changes every pixel of every L3 cutout. Because the fix depends on oversampled step-04
input, it requires a **step-04 re-run as well as step-05** (~7 h for the rung-1 config, 219
batches), followed by `calculate_snrs.py` and `_06_h5_export.py`. Landed as three separate
commits — [step04_oversampled_rendering.md](step04_oversampled_rendering.md), this, and
[subpixel_phase.md](subpixel_phase.md) — so the effect of each stays attributable even though
one rebuild covers all three. If the `weight_type` fix for the corrupted bright-core pixels
is scheduled alongside, same rule.

Affected datasets are the L3 ones — `roman_data_challenge_rung_1` and
`roman_data_challenge_rung_1_unlabeled`; both configs now set `synthetic_image.oversample: 5`.
`ella` is built with `05_galsim` (units DN), never goes through `process_batch_l3`, and
`_05_galsim.py` now rejects oversampled input outright, so it is unaffected. The rung-0
configs were left at `oversample: 1` and are unchanged until they are scheduled for a rebuild.

Note the rung-1 run used `--dither-pattern SUB6`, which is not the module default
(`BOXGAP4_1`) — the step directory records neither the level nor the pattern, so confirm the
pattern from the batch capacity (`compute_overlap_skygrid` slot count must equal the batch
size in `05_romanisim/sca*/batch_complete_*.txt`; SUB6 gives 1600 for `tile_size = 91`,
BOXGAP6_1 gives 441).

## Related

- [step04_oversampled_rendering.md](step04_oversampled_rendering.md) — the prerequisite for
  this fix, and a defect in its own right (the detector pixel response was applied twice).
- [subpixel_phase.md](subpixel_phase.md) — the L2 counterpart: every system sat at sub-pixel
  phase (0, 0) in every dataset.
- [l3_cutout_orientation.md](l3_cutout_orientation.md) — the earlier orientation/parity defect
  in the same code path; same diagnosis-and-regenerate pattern.
