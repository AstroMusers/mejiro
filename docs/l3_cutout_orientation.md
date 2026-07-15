# L3 cutout orientation: corner triangles, rotation, and mirroring

**Status:** diagnosed 2026-07-14; fix implemented in
`mejiro/pipeline/romanisim_l3_pipeline.py` (per-SCA resample rotation + lossless parity
flip at cutout extraction). Cutouts generated *before* that fix are rotated ~30°
(mod 90°) and mirror-flipped relative to their input SyntheticImages.

## Symptom

In `notebooks/view_05_romanisim_l3.ipynb`, some L3 cutouts produced by
`mejiro/pipeline/romanisim_l3_pipeline.py` show dark **triangles in their corners**, and
cutouts are **not oriented the same way** as the input SyntheticImages. Only a minority
of cutouts visibly show the triangles.

## TL;DR

The L3 mosaic comes out **north-up (east-left)**, but the L2 detector grid — where each
SyntheticImage is placed as an axis-aligned square tile — is rotated **~±60°/120° from
north** and has **opposite parity**. Every cutout (an axis-aligned square in *mosaic*
pixels) therefore contains its tile rotated by ~30° (mod 90°) and mirror-flipped. The
corner triangles are the sky-only 7-px gaps between tiles, visible only when the tile's
edge flux stands above the sky level — which is why only bright/extended systems show
them.

## Root-cause chain

### 1. The L2 detector grid is not north-aligned

With `usecrds=False`, romanisim builds the L2 WCS in the `else` branch of
`romanisim.wcs.get_wcs` via

```python
galsim.roman.getWCS(world_pos=..., SCAs=sca, date=date, PA=pa_aperture)
```

galsim's `PA` argument is the position angle of the **observatory (payload) Y axis**,
not the focal plane (`PA_is_FPA=False` by default, and romanisim does not expose it).
The WFI focal plane array is rotated by `theta_fpa = −60°` relative to the observatory
axes, and half the SCAs are mounted 180°-flipped. So `PA_APER = 0` does **not** align
detector rows with north. Measured with the pipeline's own
`mejiro.point_wfi._make_wcs` at the rung-1 pointing (RA 150°, Dec +2°, MA table 11,
2027-05-01), the position angle of the detector **+y axis** (east of north, at the
detector center) is:

| SCA | +y PA (deg) | SCA | +y PA (deg) | SCA | +y PA (deg) |
|-----|------------|-----|------------|-----|------------|
| 01  | +119.81    | 07  | +119.32    | 13  | +120.46    |
| 02  | +119.87    | 08  | +118.96    | 14  | +120.66    |
| 03  | −60.44     | 09  | −61.14     | 15  | −59.24     |
| 04  | +119.43    | 10  | +120.24    | 16  | +120.74    |
| 05  | +119.32    | 11  | +120.17    | 17  | +120.89    |
| 06  | −60.73     | 12  | −59.75     | 18  | −58.93     |

SCAs 3/6/9/12/15/18 are the 180°-flipped detectors; the ~±1° scatter across SCAs is
field-dependent distortion. In addition, the detector pixel grid's **parity** is
opposite to the FITS east-left convention (the (east, north) Jacobian of the galsim WCS
has positive determinant), i.e. the detector grid is *mirrored* relative to a standard
north-up/east-left image.

### 2. romanisim's `wcsinfo` metadata does not describe the WCS it simulated with

`romanisim.wcs.fill_in_parameters` writes

```python
roll_ref = pa_aper - V3IdlYAngle   # = 0 - (-60) = +60
```

and the defaults in `romanisim.parameters.default_parameters_dictionary` supply
`v3yangle = -60.0`, `vparity = -1` — global focal-plane placeholders, not per-SCA
values. The romanisim source itself comments *"I don't know what vparity and v3yangle
should really be"* and *"galsim.roman does something smarter with choosing the roll"*.
So the `wcsinfo` block in the saved L2 files is **inconsistent with the galsim WCS
actually used** to place pixels on the sky.

### 3. romancal resample with `rotation: None` trusts that metadata

`MosaicPipeline`'s resample step (called at `romanisim_l3_pipeline.py`,
`process_batch`) builds the output mosaic WCS through
`romancal.resample.resample_utils.make_output_wcs` →
`stcal.alignment.util.wcs_from_sregions`. With `rotation=None` it uses the *first
input's* `wcsinfo`:

```python
roll_ref = wcsinfo["roll_ref"]                      # +60°
rel_angle = roll_ref - vparity * v3yangle           # 60 - (-1)(-60) = 0
PC = [[-cos 0, sin 0], [sin 0, cos 0]] = [[-1, 0], [0, 1]]
```

→ a **north-up, east-left** mosaic. (Had the metadata described the real galsim WCS,
`rotation=None` would have produced a detector-aligned mosaic, which is its documented
intent.)

### 4. Net effect on cutouts

The tiles are squares in *detector* pixels; the mosaic is north-up. Relative to the
mosaic axes each tile is rotated by the SCA's +y PA (~+120° or ~−60°; a square's
appearance is modulo 90°, so it *looks* like ~30°) **and** mirror-flipped (parity
mismatch). `_extract_cutout` slices an axis-aligned square in mosaic pixels around the
correct sky position (`world_to_pixel` — centering is unaffected), so:

- **Corner triangles**: the cutout corners fall outside the rotated tile footprint and
  show the sky-only inter-tile gaps (`DISTORTION_GUARD = 7` px pitch margin),
  occasionally plus the bright corner of a *neighboring* tile.
- **Orientation mismatch**: the tile content is rotated ~30° (mod 90°) and reflected
  relative to the input SyntheticImage array.

### 5. Why only a minority of cutouts show triangles

The rotation affects **every** cutout identically. Visibility is brightness-selected:
the tile boundary only shows when the system's flux at the tile edge stands above the
sky background in the gaps. Verified empirically on
`05_romanisim_l3/sca01` — the brightest/most extended systems all show a crisp
~30°-rotated square boundary with dark corner triangles; median and faint systems are at
sky level at their edges and look seamless. (Per-cutout `LogNorm` percentile stretches
in the viewing notebook amplify this selection.)

## Evidence

- Diagnostic mosaic PNGs saved by the pipeline, e.g.
  `<data_dir>/<label>/05_romanisim_l3/sca01/sca01_F106_batch0_mosaic.png`: the detector
  footprint (and the tiled overlap region at its center) is a ~30°-rotated square in
  the north-up mosaic frame.
- Ranking sca01 cutouts by peak brightness: the top handful all show boundary +
  triangles; median/faint ones show none.

## Fix

Implemented in `mejiro/pipeline/romanisim_l3_pipeline.py`:

1. **Per-batch resample rotation.** In `process_batch`, the position angle of the
   dither-0 detector +y axis is measured from `wcses[0]` at the detector center and
   passed to the resample step as `rotation` (romancal `ResampleStep` spec:
   `rotation = float(default=None)  # Position angle of +y axis in degrees`). This
   makes the mosaic +y axis parallel to the detector +y axis. It must be computed per
   SCA (the angle varies by ~±1° and flips 180° for SCAs 3/6/9/12/15/18).
2. **Lossless parity flip.** `rotation` cannot change the mosaic's parity (it stays
   east-left, mirrored vs. the detector grid): with the axes made parallel, the mosaic
   +x axis is *antiparallel* to the detector +x axis. Each extracted cutout is therefore
   `np.fliplr`-ed — a lossless axis reversal, no resampling — to restore the input
   SyntheticImage orientation. The exact flip was confirmed empirically by comparing a
   bright cutout against `smooth_pixels(synth.data)` of its input pickle.
3. **Comment fix.** The `PA_APER = 0.0` comment no longer claims north alignment.
4. **`DISTORTION_GUARD` stays at 7 px.** With the mosaic detector-aligned, the residual
   tile-to-tile misalignment is only the per-dither distortion difference (a source
   lands in a different detector quadrant in each of the four BOXGAP4_1 dithers,
   ~±0.5° local rotation), which the 7-px pitch margin absorbs.

Note the fix changes future pipeline runs only; existing `.npy` cutouts must be
regenerated to get detector-aligned orientation.
