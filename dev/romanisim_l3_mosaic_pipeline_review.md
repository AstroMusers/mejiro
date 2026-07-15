# Review: romanisim → romancal `MosaicPipeline` usage in `romanisim_l3_pipeline.py`

*Review of `mejiro/pipeline/romanisim_l3_pipeline.py` (2026-07-14), verified against the
installed sources in the `mejiro-v3` conda env: **romancal 0.22.0**, **romanisim 0.13.0**.*

## Verdict

The romanisim → `MosaicPipeline` hand-off is **correct and idiomatic** for romancal
0.22.0. The metadata patching that makes romanisim L2s digestible by romancal is
exactly what is needed, and every step override is well-motivated. One deliberate
trade-off worth being aware of: cosmic rays are simulated (`crparam=dict()`) while
`outlier_detection` is skipped (details below).

## Is the romanisim → romancal interface correct?

Yes, on all four counts:

1. **L2 products.** `image.simulate(..., level=2, extra_counts=...)` returns a genuine
   L2 `ImageModel` — DN/s slopes, `err`, `var_rnoise`/`var_poisson`, DQ, and a gwcs —
   and writing `{'roman': im}` to `*_cal.asdf` is the standard romanisim output format
   that romancal opens natively.

2. **The `CAL_STEP` injection is required, not cosmetic.** romanisim never populates
   `cal_step` for L2 output (only its L3 module, `romanisim/l3.py`, does), and
   `FluxStep.apply_flux_correction` reads `model.meta.cal_step["flux"]` to decide
   whether to apply the DN/s → MJy/sr conversion. Setting `flux: 'N/A'` (rather than
   `COMPLETE`) correctly makes the conversion run; the `COMPLETE`/`N/A` values for the
   other steps satisfy the L2 schema.

3. **The `abs()` fixups are required.** romanisim's `update_photom_keywords` computes
   pixel area as a signed cross product (`sin(angle)` flips sign with WCS handedness),
   which can make `pixel_area` and hence `conversion_megajanskys` negative. Since
   `FluxStep` multiplies the data by `conversion_megajanskys`, a negative value would
   flip the sign of the whole image. The photometry is also self-consistent end to
   end: the script scales electrons using `get_abflux(band, sca)`, and romanisim's
   conversion factor is `gain·3631/abflux/10⁶/pixel_area` using the same abflux, so
   the DN/s → MJy/sr round trip closes.

4. **Association + invocation.** `asn_from_list([...], with_exptype=True)` with each
   member typed `'science'`, dumped to JSON, is the standard way to feed multiple L2s
   to the pipeline, and `MosaicPipeline.call(asn_path, ...)` is the correct entry
   point. Note this romancal version's `MosaicPipeline` runs
   `flux → skymatch → outlier_detection → resample → source_catalog`; there is no
   tweakreg (astrometric alignment), which is fine here because the WCSes are
   simulator-exact.

## The parameters, one by one

### Pipeline-level (spec: `save_results`, `on_disk`, `resample_on_skycell`)

- **`save_results=True`** — makes the pipeline write `{product_name}_coadd.asdf`;
  necessary because the script reads the mosaic back from disk via the
  `*_coadd.asdf` glob.
- **`output_dir=batch_dir`** — routes the multi-GB coadd into the per-batch working
  directory that gets `rmtree`'d after cutout extraction. Appropriate.
- **`resample_on_skycell=False`** — with the default `True`, resample and outlier
  detection would try to build the output WCS from the Roman skycell tessellation
  encoded in the association. This association has no skycell (`target=''`), so
  `False` correctly forces the output grid to be computed from the union of the four
  dither footprints at native pixel scale. (With `True` it would fall back with a
  warning; `False` is the explicit, correct choice.)
- **`on_disk`** left at its default `False` — models stay in memory, consistent with
  the script's memory strategy (few workers, `maxtasksperchild=1`).

### Step overrides

- **`skymatch: skip=True`** — SkyMatch measures and reconciles sky-level offsets
  between overlapping exposures. All four dithers are simulated with identical
  zodiacal background, so there is nothing to match; skipping saves time. Because
  romanisim L2s carry no `meta.background`, resample then treats the background as
  level 0 / already-subtracted and subtracts nothing — so the sky stays in the
  mosaic, matching the L2 sibling pipeline's cutouts. Suitable.
- **`outlier_detection: skip=True`** — this is the drizzle-median CR/outlier
  rejection across exposures. Skipping is defensible because romanisim flags
  simulated CR hits as `jump_det` per resultant and its ramp fitter
  (`fit_ramps_casertano`) excludes those resultants
  (`dq_do_not_use = saturated | jump_det`), so the L2 slopes are already largely
  CR-cleaned. **Caveat:** `crparam=dict()` in the `image.simulate` call means CRs
  *are* simulated, ramp-level rejection is not perfect, and with outlier detection
  off any residual CR artifacts propagate straight into the mosaic and cutouts. If
  pristine cutouts matter more than realism, pass `crparam=None` (no CRs) or unskip
  the step; as-is it is a reasonable speed/realism trade.
- **`source_catalog: skip=True`** — builds a source catalog and segmentation map from
  the mosaic, including PSF fitting. Irrelevant here since cutouts are extracted at
  known sky positions via the mosaic WCS; skipping avoids substantial runtime. The
  coadd is saved by the resample step regardless, so skipping the final step costs
  nothing.
- **`resample: pixel_scale_ratio=1.0`** — ratio of output to input pixel scale; 1.0
  keeps native 0.11″ Roman pixels so the L3 cutouts sample the sky identically to
  the L2 pipeline's cutouts (same `tile_size`, same format). This is the spec
  default, so passing it is explicit documentation rather than a behavior change.
- **`resample: rotation=None`** — position angle of the output +y axis; `None` means
  "inherit the mean orientation of the inputs" (vs `0.0`, which would force exact
  north-up). Since `PA_APER=0` already puts the detector rows ~north-aligned, `None`
  keeps output pixels aligned with input pixels and avoids drizzle rotation blur.
  Also the spec default — an explicit no-op.

### Resample defaults left untouched (all appropriate)

- `pixfrac=1.0` + `kernel='square'` — plain shift-and-add drizzle; shrinking pixfrac
  only pays off with a finer output grid, which is not used here.
- `weight_type='ivm'` — inverse read-noise-variance weighting; correct error
  propagation, and romanisim populates `var_rnoise`.
- `fillval='NAN'` — matches the script's NaN-padded cutout extraction.
- `good_bits='~DO_NOT_USE+NON_SCIENCE'` — `JUMP_DET` pixels remain usable, which is
  right since the ramp fit already handled them.

## Optional follow-ups (no changes required)

- Decide the CR question explicitly: `crparam=None` for clean cutouts, or unskip
  `outlier_detection` — the 4-fold coverage is exactly the regime it is designed for.
- A `pixel_scale_ratio < 1` (e.g. 0.5) would exploit the 4-point dither for
  resolution recovery, at the cost of breaking output-format parity with the L2
  pipeline — only worth it if the downstream consumer can handle a different pixel
  scale.

## Sources checked

All claims verified directly against the installed packages in
`/data/bwedig/.conda/envs/mejiro-v3`:

- romancal: `pipeline/mosaic_pipeline.py`, `flux/flux_step.py`,
  `resample/resample_step.py`, `resample/resample.py`, `skymatch`,
  `outlier_detection`, `source_catalog` step specs, `associations/asn_from_list.py`
- romanisim: `image.py` (`simulate`, `make_l2`), `l1.py` (CR injection, DQ),
  `util.py` (`update_photom_keywords`), `parameters.py` (`dq_do_not_use`)
- stpipe: `step.py` (skip semantics)
