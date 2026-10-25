# romanisim L2 products in romancal mosaics

*Behavior of `mejiro/pipeline/romanisim_l3_pipeline.py`, checked 2026-07-14 against
**romancal 0.22.0** and **romanisim 0.13.0**.*

## L2-to-L3 handoff

Metadata updates let romancal read romanisim L2 products. The step overrides match
this simulated-data workflow. Cosmic rays are simulated with `crparam=dict()`, and
`outlier_detection` is skipped. Residual hits can therefore remain in the mosaic.

1. **L2 products.** `image.simulate(..., level=2, extra_counts=...)` returns a genuine
   L2 `ImageModel` — DN/s slopes, `err`, `var_rnoise`/`var_poisson`, DQ, and a gwcs —
  Writing `{'roman': im}` to `*_cal.asdf` produces the standard romanisim output
  format, which romancal opens natively.

2. **The `CAL_STEP` metadata controls flux conversion.** romanisim populates
  `cal_step` for L3 output in `romanisim/l3.py`, but leaves it empty for L2 output.
  `FluxStep.apply_flux_correction` reads `model.meta.cal_step["flux"]` before
  converting DN/s to MJy/sr. Setting `flux: 'N/A'` lets the conversion run. The
  `COMPLETE` and `N/A` values assigned to other steps satisfy the L2 schema.

3. **The `abs()` fixups preserve the image sign.** romanisim's
  `update_photom_keywords` computes pixel area as a signed cross product. The sign
  of `sin(angle)` changes with WCS handedness, which can make `pixel_area` and
  `conversion_megajanskys` negative. `FluxStep` multiplies the data by that
  conversion factor, so a negative value would reverse the image sign. The script
  scales electrons with `get_abflux(band, sca)`. romanisim uses the same `abflux` in
  `gain·3631/abflux/10⁶/pixel_area`, so the DN/s-to-MJy/sr round trip closes.

4. **Association + invocation.** `asn_from_list([...], with_exptype=True)` with each
  member typed `'science'` and dumped to JSON feeds multiple L2 products to the
  pipeline. `MosaicPipeline.call(asn_path, ...)` is its entry point. In romancal
  0.22.0, the steps run in this order: `flux → skymatch → outlier_detection →
  resample → source_catalog`. The pipeline has no `tweakreg` astrometric-alignment
  step. The simulated WCS solutions are exact, so alignment is unnecessary here.

## Pipeline settings

### Pipeline-level settings for `save_results`, `on_disk`, and `resample_on_skycell`

- **`save_results=True`** — writes `{product_name}_coadd.asdf`. The script reads
  the mosaic back from disk using the `*_coadd.asdf` glob.
- **`output_dir=batch_dir`** — routes the multi-GB coadd into the per-batch working
  directory. The script removes that directory after cutout extraction.
- **`resample_on_skycell=False`** — with the default `True`, resample and outlier
  detection build the output WCS from the Roman skycell tessellation in the
  association. This association has no skycell (`target=''`). Setting this option
  to `False` builds the output grid from the four dither footprints at native pixel
  scale. The `True` setting falls back with a warning.
- **`on_disk`** left at its default `False` — models stay in memory, consistent with
  the script's use of few workers and `maxtasksperchild=1`.

### Step overrides

- **`skymatch: skip=True`** — SkyMatch measures sky-level offsets between overlapping
  exposures. The four dithers use the same simulated zodiacal background, so
  skipping this step saves time. romanisim L2 products have no `meta.background`,
  so resample treats the background as zero and subtracts nothing. The mosaic keeps
  the sky, as do the L2 sibling-pipeline cutouts.
- **`outlier_detection: skip=True`** — this is the drizzle-median CR/outlier
  rejection across exposures. Skipping is defensible because romanisim flags
  simulated CR hits as `jump_det` per resultant. Its ramp fitter,
  `fit_ramps_casertano`, excludes those resultants with
  `dq_do_not_use = saturated | jump_det`, so the L2 slopes are largely CR-cleaned.
  The simulation still includes CRs through `crparam=dict()`, and ramp-level
  rejection is imperfect. With outlier detection off, residual artifacts can reach
  the mosaic and cutouts. Set `crparam=None` to omit CRs or enable outlier detection
  to reject residual hits across exposures.
- **`source_catalog: skip=True`** — builds a source catalog and segmentation map from
  the mosaic, including PSF fitting. Cutouts use known sky positions and the mosaic
  WCS, so the catalog is unnecessary. Skipping this step avoids substantial
  runtime. The resample step still saves the coadd.
- **`resample: pixel_scale_ratio=1.0`** — sets the output scale equal to the input
  scale. This keeps the native 0.11″ Roman pixel scale, so L3 cutouts match the L2
  pipeline's `tile_size` and sampling. This is the specification default.
- **`resample: rotation=None`** — uses the mean input orientation for the output
  +y axis. A value of `0.0` forces exact north-up. Since `PA_APER=0` already aligns
  detector rows roughly north, `None` keeps output and input pixels aligned and
  avoids drizzle rotation blur. This is also the specification default.

### Resample defaults

- `pixfrac=1.0` + `kernel='square'` — shift-and-add drizzle. A smaller pixfrac is
  useful with a finer output grid, which this workflow does not use.
- `weight_type='ivm'` — inverse read-noise-variance weighting. romanisim populates
  `var_rnoise`, which supports error propagation.
- `fillval='NAN'` — matches the script's NaN-padded cutout extraction.
- `good_bits='~DO_NOT_USE+NON_SCIENCE'` — keeps `JUMP_DET` pixels usable because the
  ramp fit has already rejected those resultants.

## Alternative choices

- Setting `crparam=None` omits simulated cosmic rays. Enabling
  `outlier_detection` can reject residual hits across the four exposures.
- A `pixel_scale_ratio < 1`, such as 0.5, can recover resolution from the four-point
  dither. It changes the output scale relative to the L2 pipeline, so downstream
  consumers must support the higher-resolution cutouts.
