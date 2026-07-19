# Future work: INTERPOL deflection-map optimization for step 04

The pipeline previously had a third step-04 variant,
`_04_create_synthetic_images_interpol.py`, deleted when the step-04 scripts
were consolidated into `_04_create_synthetic_images.py` (2026-07). This
document preserves the technique so it can be reimplemented against the
consolidated script in a future session. The full original script is
recoverable from git history (`git log --diff-filter=D -- '*interpol*'`).

## Rationale

The deflection field depends only on the mass model and is band-independent.
Instead of ray-tracing the full lens model (NFW, Hernquist, thousands of
subhalos, ...) independently for each photometric band, evaluate the lens model
**once** on a fine grid, package the result as lenstronomy's `INTERPOL` lens
profile, and reuse that interpolated map for every band. For heavily
substructured systems rendered in several bands, this removes the dominant
redundant cost.

## Proposed integration

Add a config knob under `synthetic_image:`, e.g.:

```yaml
synthetic_image:
  interpol: True
```

and in `create_synthetic_image()` in `_04_create_synthetic_images.py`, after
unpickling the lens: compute the map, swap the lens model to `INTERPOL`, run
the per-band loop unchanged, and restore afterwards. The swap/restore sequence
from the original script:

```python
# --- Swap lens model to INTERPOL ---
original_lens_model_list = lens.lens_model_list
original_kwargs_lens = lens.kwargs_lens
original_use_jax = lens.use_jax

lens.lens_model_list = ['INTERPOL']
lens.kwargs_lens = [interpol_kwargs]
lens.use_jax = [False]

# ... per-band SyntheticImage loop ...

# --- Restore original lens model ---
lens.lens_model_list = original_lens_model_list
lens.kwargs_lens = original_kwargs_lens
lens.use_jax = original_use_jax
```

Note the interaction with the JAX path: the consolidated script's
`_enable_jax_on_lens()` sets `use_jax` per profile; `INTERPOL` is evaluated
with plain lenstronomy (`use_jax = [False]`), so `interpol: True` and
`use_jax: True` are effectively mutually exclusive per system unless
jaxtronomy grows an INTERPOL profile.

## Key functions from the deleted script

```python
def _compute_interpol_deflection_map(lens, fov_arcsec, pixel_scale, supersampling_factor):
    """Compute the deflection map on a fine grid and return INTERPOL kwargs.

    Evaluates the full lens model (alpha, potential, hessian) on a supersampled
    grid and packages the results as a dictionary suitable for lenstronomy's
    INTERPOL lens profile. pixel_scale should be the finest scale across bands.
    """
    num_pix_grid = util.set_odd_num_pix(fov_arcsec, pixel_scale)
    adjusted_fov = num_pix_grid * pixel_scale
    num_pix_interp = num_pix_grid * supersampling_factor

    interp_coords = np.linspace(-adjusted_fov / 2, adjusted_fov / 2, num_pix_interp)
    xx, yy = np.meshgrid(interp_coords, interp_coords)
    x_flat, y_flat = xx.ravel(), yy.ravel()

    lens_model = lens.lens_model
    kwargs_lens = lens.kwargs_lens

    alpha_x, alpha_y = lens_model.alpha(x_flat, y_flat, kwargs_lens)
    try:
        potential = lens_model.potential(x_flat, y_flat, kwargs_lens)
    except ValueError:
        potential = np.zeros_like(x_flat)
    f_xx, f_xy, _, f_yy = lens_model.hessian(x_flat, y_flat, kwargs_lens)

    shape = (num_pix_interp, num_pix_interp)
    return {
        'grid_interp_x': interp_coords,
        'grid_interp_y': interp_coords,
        'f_': potential.reshape(shape),
        'f_x': alpha_x.reshape(shape),
        'f_y': alpha_y.reshape(shape),
        'f_xx': f_xx.reshape(shape),
        'f_yy': f_yy.reshape(shape),
        'f_xy': f_xy.reshape(shape),
    }
```

Usage in the worker:

```python
pixel_scales = {b: pipeline.instrument.get_pixel_scale(b).value for b in bands}
finest_pixel_scale = min(pixel_scales.values())
interpol_kwargs = _compute_interpol_deflection_map(
    lens, fov_arcsec, finest_pixel_scale, supersampling_factor
)
lens.interpol_deflection_map = interpol_kwargs  # store on lens for persistence
```

Adaptive supersampling needs the grid pre-computed from the *original* lens
model, because `SyntheticImage.build_adaptive_grid()` reads
`center_x`/`center_y` from `kwargs_lens[0]`, which INTERPOL kwargs lack:

```python
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Util import util as lenstronomy_util

def _precompute_adaptive_grid(lens, band, instrument, fov_arcsec, pad=40):
    """Pre-compute the adaptive supersampling grid for a given band.

    Replicates SyntheticImage.build_adaptive_grid() using the original
    (non-INTERPOL) lens model. Returns a boolean mask for supersampled pixels.
    """
    pixel_scale = instrument.get_pixel_scale(band).value
    num_pix = util.set_odd_num_pix(fov_arcsec, pixel_scale)

    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = (
        lenstronomy_util.make_grid_with_coordtransform(
            numPix=num_pix, deltapix=pixel_scale, subgrid_res=1,
            left_lower=False, inverse=False))
    coords = Coordinates(Mpix2coord, ra_at_xy_0, dec_at_xy_0)

    image_x, image_y = lens.get_image_positions()
    img_pix_x, img_pix_y = coords.map_coord2pix(ra=image_x, dec=image_y)

    image_radii = []
    for x, y in zip(img_pix_x, img_pix_y):
        image_radii.append(np.sqrt((x - (num_pix // 2)) ** 2 + (y - (num_pix // 2)) ** 2))

    x = np.linspace(-num_pix // 2, num_pix // 2, num_pix)
    y = np.linspace(-num_pix // 2, num_pix // 2, num_pix)
    X, Y = np.meshgrid(x, y)
    center_x = lens.kwargs_lens[0].get('center_x', 0)
    center_y = lens.kwargs_lens[0].get('center_y', 0)
    distance = np.sqrt((X - (center_x / pixel_scale)) ** 2 + (Y - (center_y / pixel_scale)) ** 2)

    min_r = max(np.min(image_radii) - pad, 0)
    max_r = min(np.max(image_radii) + pad, num_pix // 2)

    return (distance >= min_r) & (distance <= max_r)
```

The pre-computed mask is passed per band via
`kwargs_numerics['supersampled_indexes']` when
`supersampling_compute_mode == 'adaptive'`.

## What a reimplementation must add

The deleted script predated several features of the consolidated step 04; a
reimplementation must support all of them:

- `--resume` / `_is_complete` skip logic and the delete-unless-resume block
- `serialization: full | lightweight` (`.pkl` vs `.npz` via `save_lightweight`)
- `deflector_only_fraction` handling
- the `.psfpos.json` sidecar written per output for `_05_romanisim` PSF
  bucketing
- `--prev-step {02,03}` input selection
- `require_cached: True` in the PSF kwargs (the old script omitted it)
- stale `failed_*.pkl` cleanup before retrying a band
