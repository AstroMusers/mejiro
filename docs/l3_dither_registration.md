# L3 dither registration: integer tile placement misregisters the dithers

**Status:** diagnosed 2026-07-26; **not fixed**. Every L3 dataset produced by
`mejiro/pipeline/_05_romanisim.py --level l3` to date is affected, including
`roman_data_challenge_rung_1` v3.0. Found while investigating a separate defect (corrupted
bright-core pixels from the resample `ivm` weighting); deliberately kept out of that fix so
the two changes stay separately attributable.

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

## Proposed fix

The tile must be rendered at its true sub-pixel position, which means `_place_tile` can no
longer be a plain slice assignment. Options, roughly in order of preference:

1. **Shift the tile sub-pixel before placement.** Keep the integer slice, but pre-shift the
   electron tile by the fractional residual (`src_pix.x - round(src_pix.x)`, likewise y)
   before placing it. A band-limited shift (`scipy.ndimage.shift` with a high spline order, or
   an FFT phase shift) preserves total flux to numerical precision. Cheap: one shift per
   system per dither, on a 91x91 array. Watch for ringing at the tile edges — the tiles are at
   sky level at their boundaries, so `mode='nearest'` or a small taper should be adequate, and
   flux conservation should be asserted.
2. **Render through galsim at the true position.** Draw the tile as a galsim image with an
   explicit sub-pixel offset (`drawImage(..., offset=...)`), which handles the interpolation
   with the machinery already imported. More invasive; the tiles are currently plain numpy
   arrays derived from `smooth_pixels(synth.data)`.
3. **Snap the sky grid so every dither rounds to zero.** Not possible in general: the dithers
   have different distortion, so no single sky position is integer in all of them.

Option 1 is the smallest change consistent with the file's existing structure.

Whichever is chosen, `DISTORTION_GUARD = 7` and the overlap-grid logic are unaffected — this
changes only where within its cell a tile is rendered, not the cell layout.

## Validation

- Assert flux conservation on the shifted tile (`sum` before vs after) to the float64 level.
- Re-measure the residuals: after the fix, `wcses[d].toImage(sky)` residuals should be fully
  absorbed by the shift, so the co-add should retain ~1.000 of the correctly-registered peak
  instead of 0.972.
- Confirm the commanded sub-pixel pattern actually survives to the mosaic: with SUB6 the six
  renderings should land at the six commanded phases, not at six rounding residuals.
- Compare a bright system's co-added cutout against `smooth_pixels(synth.data)` of its step-04
  input, as was done for the parity flip in
  [l3_cutout_orientation.md](l3_cutout_orientation.md) — the core should sharpen.
- The pytest pipeline test (`tests/test_pipeline/test_mejiro_pipeline.py`, `test.yaml`, ~68 s,
  `data_dir: null`) must still pass. Do not validate by running a real project config —
  `_05_romanisim.py` deletes all output in `output_dir` without `--resume`.

## Impact / regeneration

This changes every pixel of every L3 cutout, so it requires a full step-05 re-run (~7 h for
the rung-1 config, 219 batches) followed by `calculate_snrs.py` and `_06_h5_export.py`. If it
is scheduled alongside the `weight_type` fix for the corrupted bright-core pixels, the two
should still be landed as separate commits so the effect of each is attributable, even if
only one rebuild is run.

Affected datasets are the L3 ones — `roman_data_challenge_rung_1` and
`roman_data_challenge_rung_1_unlabeled`. `ella` is built with `05_galsim` (units DN) and never
goes through `process_batch_l3`, so it is unaffected.

Note the rung-1 run used `--dither-pattern SUB6`, which is not the module default
(`BOXGAP4_1`) — the step directory records neither the level nor the pattern, so confirm the
pattern from the batch capacity (`compute_overlap_skygrid` slot count must equal the batch
size in `05_romanisim/sca*/batch_complete_*.txt`; SUB6 gives 1600 for `tile_size = 91`,
BOXGAP6_1 gives 441).

## Related

- [l3_cutout_orientation.md](l3_cutout_orientation.md) — the earlier orientation/parity defect
  in the same code path; same diagnosis-and-regenerate pattern.
