# Understanding L3 Cutout Orientation

Roman images are resampled into a north-up mosaic before cutouts are extracted. Each
detector has its own rotation and parity, so an extracted tile can appear rotated and
mirrored relative to its input `SyntheticImage`. The pipeline applies a per-SCA rotation
and restores the input parity during cutout extraction. Existing `.npy` cutouts retain
their previous orientation and must be regenerated.

## Symptom

In `notebooks/view_05_romanisim_l3.ipynb`, some L3 cutouts produced by
`mejiro/pipeline/_05_romanisim.py` show dark **triangles in their corners**, and
cutouts are **not oriented the same way** as the input SyntheticImages. Only a minority
of cutouts visibly show the triangles.

## Summary

The L3 mosaic comes out **north-up (east-left)**, but the L2 detector grid – where each
SyntheticImage is placed as an axis-aligned square tile – is rotated **~±60°/120° from
north** and has **opposite parity**. Every cutout (an axis-aligned square in *mosaic*
pixels) therefore contains its tile rotated by ~30° (mod 90°) and mirror-flipped. The
corner triangles are the sky-only 7-px gaps between tiles, visible only when the tile's
edge flux stands above the sky level – which is why only bright/extended systems show
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

SCAs 3/6/9/12/15/18 are flipped by 180°. The remaining ~1° variation among SCAs comes
from field-dependent distortion. The detector pixel grid's **parity** is
opposite to the FITS east-left convention (the (east, north) Jacobian of the galsim WCS
has positive determinant), i.e. the detector grid is *mirrored* relative to a standard
north-up/east-left image.

### 2. romanisim's `wcsinfo` metadata does not describe the WCS it simulated with

`romanisim.wcs.fill_in_parameters` writes

```python
roll_ref = pa_aper - V3IdlYAngle   # = 0 - (-60) = +60
```

and the defaults in `romanisim.parameters.default_parameters_dictionary` supply
`v3yangle = -60.0`, `vparity = -1` – global focal-plane placeholders, not per-SCA
values. The romanisim source itself comments *"I don't know what vparity and v3yangle
should really be"* and *"galsim.roman does something smarter with choosing the roll"*.
So the `wcsinfo` block in the saved L2 files is **inconsistent with the galsim WCS
actually used** to place pixels on the sky.

### 3. romancal trusts its metadata when rotation is unset

`MosaicPipeline`'s resample step (called at `_05_romanisim.py`,
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

The tiles are squares in *detector* pixels. The mosaic is north-up. Each tile follows
its SCA's +y position angle, about +120° or −60°. Because a square looks the same after
a 90° rotation, the misalignment appears to be about 30°. Its parity is also reversed.
`_extract_cutout` slices an axis-aligned square in mosaic pixels around the
correct sky position (`world_to_pixel` – centering is unaffected), so:

- **Corner triangles.** Cutout corners can fall outside the rotated tile footprint and
   show sky in the inter-tile gaps (`DISTORTION_GUARD = 7` px), sometimes with a bright
   corner from a neighboring tile.
- **Orientation mismatch.** Tile content is rotated by roughly 30° modulo 90° and
   reflected relative to the input `SyntheticImage` array.

### 5. Why only a minority of cutouts show triangles

The rotation affects **every** cutout identically. The boundary appears only when the
system's flux at the tile edge rises above the sky background in the gaps. In
`05_romanisim/sca01`, the brightest and most extended systems show a crisp, ~30°-rotated
square boundary with dark corner triangles. Median and faint systems have edge flux near
the sky level, so their boundaries blend into the background. Per-cutout `LogNorm`
percentile stretches in the viewing notebook make this brightness selection more visible.

## Evidence

- The mosaic at `<data_dir>/<label>/05_romanisim/sca01/sca01_F106_batch0_mosaic.png` shows the detector footprint and central tiled-overlap region as a roughly 30°-rotated square in the north-up frame.
- Ranking sca01 cutouts by peak brightness shows the boundary and corner triangles in the brightest few. Median and faint cutouts show neither.

## Detector-Aligned Cutouts

1. **Per-SCA resampling rotation.** In `process_batch`, measure the position angle of
   the dither-0 detector +y axis from `wcses[0]` at the detector center. Pass that angle
   as `rotation` to the resampling step. This aligns mosaic +y with detector +y. The
   angle varies by about 1° and flips by 180° for SCAs 3/6/9/12/15/18, so compute it
   separately for each SCA.
2. **Lossless parity flip.** Rotation preserves the mosaic's east-left parity. After
   aligning the +y axes, mosaic +x points opposite detector +x. Apply `np.fliplr` to each
   extracted cutout to restore the input `SyntheticImage` orientation. This axis reversal
   does not resample pixels. The validator preserves finite, nonnegative pixels and
   rejects invalid model output without filling it.
3. **Distortion margin.** Keep `DISTORTION_GUARD` at 7 px. After detector alignment, the
   remaining tile mismatch comes from per-dither distortion differences. A source falls
   in a different detector quadrant in each BOXGAP4_1 dither, with roughly 0.5° of local
   rotation. The 7-px pitch margin absorbs this difference.

Existing `.npy` cutouts keep their original orientation. Regenerate them to apply the
detector-aligned orientation.
