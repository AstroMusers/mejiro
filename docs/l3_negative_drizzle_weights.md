# L3 co-add: float16 variance reconstruction produces negative drizzle weights

**Status:** diagnosed and confirmed by reproduction 2026-07-26; fix implemented in
`mejiro/pipeline/_05_romanisim.py` (`process_batch_l3` now passes `weight_type='exptime'` to
the resample step). Cutouts generated *before* that fix contain corrupted pixels in the cores
of the brightest systems and must be regenerated.

## Symptom

In `notebooks/view_06.ipynb`, `uid=00000005` / F129 shows a 236.3 MJy/sr pixel at the
deflector core where the same pixel in F106/F158 reads ~31/42 and the F129/F158 ratio is a
flat 0.8 everywhere else in the frame.

`mejiro.utils.qa.find_corrupted_exposures` over `roman_data_challenge_rung_1` v3.0 (107,507
systems x 3 bands = 322,521 exposures) finds **246 corrupted exposures across 199 systems
(0.185%)**, including **133 exposures containing negative surface brightness**, down to
**-9387.5 MJy/sr**. Worst cases:

| uid | band | artifact |
|-----|------|----------|
| 00105272 | F158 | 5328.8 next to -0.1, neighbours ~12 |
| 00019631 | F106 | 432.8 next to 0.1 |
| 00019631 | F158 | -176.3 and -38.4 |
| 00001004 | all three | -435, -26, -560 |

A negative value cannot be produced by any additive step in this pipeline: the co-add is a
positive-definite surface brightness. Something was dividing by a weight that passes through
zero.

## TL;DR

romanisim's L2 files carry **no `var_rnoise` array**, and store `err` and `var_poisson` as
**float16**. romancal reconstructs the read-noise variance it needs for `ivm` weighting as
`err.astype(float32)**2 - var_poisson`. On bright pixels the read-noise term is a tiny
fraction of the total variance, so that subtraction is catastrophic cancellation between two
three-significant-digit numbers: above ~100 DN/s the result is *entirely* float16
quantization noise, and **19% of those pixels come out negative**.

`stcal` inverts it to get the drizzle weight and discards only *non-finite* results, so a
negative variance passes straight through as a **negative weight**. Drizzle's
`sum(w*d)/sum(w)` then has a denominator that can pass through zero, and the output pixel
explodes — positive, near-zero, or negative.

Switching the resample step to `weight_type='exptime'` bypasses the reconstruction entirely.

## Root-cause chain

### 1. romanisim's L2 has no `var_rnoise`, and float16 variances

`romanisim.image.make_asdf` computes `var_rnoise` from the ramp fit and passes it to
`out.update(...)`, but `roman_datamodels`' `WfiImage` schema has no such field, so the value
is silently dropped. Inspecting a real L2 written by `process_batch_l3`:

```
top-level keys: amp33, border_ref_pix_*, chisq, data, dq, dumo, err, meta, var_poisson
  data:        (4088, 4088) float32
  dq:          (4088, 4088) uint32
  err:         (4088, 4088) float16     <-- three significant digits
  var_poisson: (4088, 4088) float16     <-- three significant digits
```

`make_asdf` casts each field to `out[field].dtype`, i.e. to whatever the schema declares, so
the float16 storage is not something the pipeline chooses.

### 2. romancal reconstructs it by subtraction

`romancal.resample.resample` line ~215 builds its model dict with
`"var_rnoise": compute_var_rnoise(model)`, and
`romancal.lib.basic_utils.compute_var_rnoise` is:

```python
if hasattr(model, "var_rnoise"):
    return model.var_rnoise
var_rnoise = model.err.astype(np.float32) ** 2
for var_name in ["var_poisson", "var_flat", "var_dark"]:
    if hasattr(model, var_name):
        var_rnoise -= getattr(model, var_name)
return var_rnoise
```

`hasattr(model, 'var_rnoise')` is False here (it raises `AttributeError: No such attribute
(var_rnoise) found in node: WfiImage`), so the subtraction branch always runs.

### 3. The subtraction is catastrophic cancellation on bright pixels

`err**2 = var_rnoise + var_poisson` exactly, but only to float16 precision once stored. The
relative quantization step of float16 is ~2^-11, so the absolute error in the difference is
~`5e-4 * var_poisson`. The true `var_rnoise` is ~4e-5 (DN/s)^2 and is flux-independent, so
once `var_poisson` exceeds ~0.1 the quantization error exceeds the quantity being measured.

Measured on the reproduced sca01/F129 L2 (dither 0), binning all 16.7M pixels by brightness:

| data (DN/s) | n | median reconstructed `var_rnoise` | fraction < 0 | fraction below the float16 quantization step |
|-------------|---|-----------------------------------|--------------|---------------------------------------------|
| 0 - 1     | 16,562,448 | 2.07e-05 | 0.0%  | 0%   |
| 1 - 5     | 132,810    | 2.1e-05  | 0.0%  | 0%   |
| 5 - 20    | 13,470     | 3.9e-05  | 0.0%  | ~0%  |
| 20 - 50   | 2,340      | 4.25e-05 | 0.0%  | 32%  |
| 50 - 100  | 531        | 3.43e-05 | **11.1%** | 88%  |
| 100 - 200 | 132        | 7.47e-05 | **18.9%** | **100%** |

All 86 pixels on the detector with a negative reconstructed `var_rnoise` have data > 50 DN/s.
That is the entire brightness selection: the defect can only touch bright cores.

### 4. A negative variance becomes a negative drizzle weight

`stcal.resample.utils._get_inverse_variance`:

```python
inv = 1.0 / array
inv[~np.isfinite(inv)] = 0   # zeros for bad pixels
```

Only non-finite reciprocals are discarded. `1/(-2.3e-4)` is perfectly finite, so it survives.
The weights across the `uid=00000005` core in the reproduced L2, adjacent pixels:

```
   4096.0   127100.1     2621.4     3938.3    -8256.5
   2945.4    -4301.9     2593.9     3039.4   -11397.6
 -17476.3      947.9     1087.7     4190.1   -10951.2
   1820.4     1858.4     6384.0     2255.0     4599.0
   4660.3     6887.2     6543.4   -13148.3    34663.7
```

Sign-indefinite and swinging by more than an order of magnitude between neighbours — this is
noise, not a weight map.

### 5. Drizzle divides by a denominator that can cross zero

The output pixel is `sum(w*d)/sum(w)` accumulated in float32 over the six dithers. With
mixed-sign weights the denominator can land arbitrarily near zero, and the quotient explodes
in either direction. That is the 5328.8, the -0.1, and the -9387.5.

`outlier_detection` is skipped in this pipeline (for speed; the co-add has no real CRs to
reject), so nothing downstream caught it.

### 6. The sub-pixel dither pattern removes the only escape

`roman_data_challenge_rung_1` v3.0 was produced with `--dither-pattern SUB6` (confirmed by
matching `compute_overlap_skygrid`'s 1600-slot capacity against the batch sizes in
`05_romanisim/sca*/batch_complete_*.txt`). A sub-pixel pattern puts every system on
essentially the same detector pixels in all six exposures, so a bright core pixel gets a
garbage weight in *all six* at once. A gap-filling pattern would land the system on different
pixels each time and would partially average the damage away; it would not remove it.

## What was ruled out

- **Step 04.** The `SyntheticImage` npz files for the affected systems are smooth,
  single-peaked and outlier-free. Verified directly.
- **The h5 export.** `_06_h5_export.py` writes `util.load_exposure(...).data` verbatim from
  `05_romanisim/sca*/Exposure_*.npy`; the corruption is already in the `.npy`.
- **Cosmic rays.** `crparam=dict()` does enable CRs (~99,000 hits per detector per 736.86 s
  ramp: `flux 8 cm^-2 s^-1 * area 16.8 cm^2 * 233 reads * read_time`), and
  `outlier_detection` is skipped so nothing rejects them. But re-running `l1.make_l1` +
  `image.make_l2` on the actual uid-5 tile with and without CRs changes the L2 slope by at
  most 3%: romanisim flags the jump resultant and `fit_ramps_casertano` drops it. CR hits are
  also spatially uniform, which contradicts the 97.1%-within-5-px-of-centre concentration of
  the corrupted pixels.
- **The L2 slopes.** Reconstructing ramps from 10 to 12,800 DN/s (up to 170x saturation), the
  fitted slope stays accurate to <1% down to 2 usable resultants. It only collapses to
  exactly 0 at <=1 usable, which needs a ~9.4M DN ramp — far beyond anything in this dataset.
  The reproduced L2 confirms this: the `uid=00000005` core reads 236-270 DN/s with a smooth
  profile, and only becomes garbage after the co-add.
- **Ramp-fit variance inflation (the first hypothesis, and wrong).** Saturated resultants
  *are* dropped from the fit — the reproduced L2 shows `dq == 2` (SATURATED) across the whole
  `uid=00000005` core — and romanisim's own `var_rnoise` does inflate by up to ~10^3 as a
  result. But that array never reaches romancal (step 1), so it cannot be the mechanism.
  Saturation matters here only because it marks which pixels are bright.

## Fix

In `process_batch_l3`, the resample step is now called with an explicit weighting:

```python
'resample': {'pixel_scale': _mosaic_pixel_scale(l2_files[0]),
             'rotation': _detector_y_pa_deg(wcses[0]),
             'weight_type': 'exptime'},
```

`build_driz_weight` then uses `exposure_time * dqmask` — one uniform, strictly positive weight
per dither — and never calls `compute_var_rnoise` at all. All dithers come from the same MA
table at the same depth, so a uniform per-exposure weight is the correct choice here
independently of this bug. Because the L2 slopes are sound, this restores correct core values
rather than merely suppressing a symptom.

### Confirmed by reproduction

`sca01/F129/batch0` (which contains both `uid=00000005` and `uid=00001004`) was replayed
exactly as production ran it — same 1600 tiles, same grid slots, same seeds, same SUB6
pattern — and the six resulting L2s were then co-added **twice, varying only `weight_type`**.

The `ivm` co-add reproduces the shipped cutouts bit for bit, which validates the replay:

```
uid 00000005 F129, rows 41-47 x cols 40-47
  shipped production        repro ivm                 repro exptime (FIXED)
  22.42 25.52 29.36 236.27  22.42 25.52 29.36 236.27  22.28 26.03 30.67 35.67
  22.37 27.32 27.78  62.54  22.37 27.32 27.78  62.54  22.53 26.89 32.85 40.61
  20.90 24.98 29.24  32.67  20.90 24.98 29.24  32.67  21.05 24.98 30.15 36.80
```

Summary over the full 91x91 cutouts:

| | data min | data max | data n<0 | weight min | weight max | weight n<0 |
|---|---|---|---|---|---|---|
| uid 00000005, shipped | 0.3457 | 236.3 | 0 | — | — | — |
| uid 00000005, repro `ivm` | 0.3457 | 236.3 | 0 | -1.702e+06 | 1.333e+06 | 3 |
| uid 00000005, repro `exptime` | 0.3465 | **51.33** | 0 | 4421 | 4421 | **0** |
| uid 00001004, shipped | -25.95 | 777.9 | 5 | — | — | — |
| uid 00001004, repro `ivm` | -25.95 | 777.9 | 5 | -1.149e+07 | 4.415e+07 | 262 |
| uid 00001004, repro `exptime` | **0.5776** | **194.5** | **0** | 4421 | 4421 | **0** |

The negative drizzle weights predicted in step 4 are present in the mosaic (262 of them for
`uid=00001004`). With `exptime` the weight is a constant 4421 (the co-add exposure time),
the spikes and negatives are gone, the cores are smooth single-peaked profiles again, and the
background is unchanged (0.3457 -> 0.3465) — the fix touches only what was broken.

Alternatives considered and rejected:

- **Write a real `var_rnoise` into the L2.** The schema has no field for it, so it would mean
  either carrying a romanisim patch or writing `err` in float32 — neither is available from
  the pipeline side, and `ivm` would still be the wrong weighting for six equal-depth
  exposures.
- **Set DO_NOT_USE on saturated pixels** so `good_bits` excludes them. This addresses a
  different thing (romanisim sets SATURATED/JUMP_DET but never DO_NOT_USE, so `good_bits`
  never masks anything), and since saturation is deterministic the pixel would be excluded
  from every dither, leaving `fillval='NAN'` holes. It also discards slopes that are accurate.
- **Enable `outlier_detection`.** The artifact is deterministic and near-identical across
  dithers, so there is no outlier to reject. It also costs significant time.

## Regeneration

The fix changes future pipeline runs only. Existing L3 `.npy` cutouts — and every h5 built
from them — must be regenerated: step 05, then `calculate_snrs.py`, then `_06_h5_export.py`.
The change alters the weighting of *every* pixel, not just the corrupted ones, so a partial
re-run of only the affected batches would leave the dataset internally inconsistent.

Affected datasets are those built through `process_batch_l3` — identified by a `05_romanisim`
step directory whose `exposure_level.txt` reads `l3`, equivalently by `units = MJy/sr` on the
exported exposures:

| dataset | step 05 | units | affected |
|---------|---------|-------|----------|
| `roman_data_challenge_rung_1` v3.0 | `05_romanisim` (l3) | MJy/sr | **yes** — 246 exposures / 199 of 107,507 systems (0.185%) |
| `roman_data_challenge_rung_1_unlabeled` v3.0 | `05_romanisim` (l3) | MJy/sr | **yes** — 200 exposures / 149 of 95,603 systems (0.156%) |
| `ella` | `05_galsim` | DN | no — never went through the co-add; scanned clean (0 of 4,376) |

The two L3 datasets are corrupted at the same rate, as expected for a defect driven by the
brightness distribution of the deflector population rather than by anything run-specific.

## Detecting it

`mejiro.utils.qa.find_corrupted_exposures` flags an exposure containing a non-finite pixel, a
negative pixel, or a pixel outside `[0.5, 5.0]` times the median of its eight neighbours. It
runs over the full 322,521-exposure dataset in ~9 minutes and selects 246 exposures / 199
systems on `rung_1` v3.0.

Those thresholds are set from the dataset's own clean baseline: over the 290,268 exposures
faint enough that they cannot be affected (bottom 90% by peak surface brightness), the
neighbour ratio spans [0.79, 3.30], so [0.5, 5.0] leaves ~1.5x headroom on both sides and
still catches every corrupted system. `_06_h5_export.py` calls `qa.check_exposures`, which
raises, whenever its input came from a romanisim step.

## Related

- [l3_cutout_orientation.md](l3_cutout_orientation.md) — earlier orientation/parity defect in
  the same code path.
- [l3_dither_registration.md](l3_dither_registration.md) — a second, still-unfixed defect
  found during this investigation: integer tile placement misregisters the dithers.
