# pyHalo: duplicate lens-plane redshifts cause `R = 0.00e+00 is too small`

## Symptom

`pyHalo.preset_models.CDM(...)` (and any other preset that includes the two-halo
term) raises:

```
Exception: R = 0.00e+00 is too small (min. R = 1.00e-03)
```

from `colossus.cosmology.cosmology.Cosmology.correlationFunction` for a small
fraction (~0.1%) of `(z_lens, z_source)` pairs. The traceback runs through
`pyHalo/Rendering/two_halo.py::two_halo_enhancement_factor` →
`scipy.integrate.quad(_boost_integrand, rmin, rmax, ...)`.

## Root cause

`pyHalo/Rendering/two_halo.py::two_halo_enhancement_factor` computes

```python
rmax = lens_cosmo.cosmo.D_C_transverse(z_lens + z_step) - lens_cosmo.cosmo.D_C_transverse(z_lens)
rmin = min(rmax, 0.5)
two_halo_boost = 2 * quad(_boost_integrand, rmin, rmax, args=args)[0] / (rmax - rmin)
```

where `z_step` is the redshift spacing at the host plane, taken from
`delta_z_list[idx]` produced by `pyHalo.utilities.generate_lens_plane_redshifts`.

When `z_step == 0`, `rmax == 0`, `rmin == 0`, and `scipy.quad` evaluates the
integrand at `r = 0`. The integrand passes `r * h` to colossus's
`correlationFunction(R, z)`, which rejects any `R < 1e-3 Mpc/h`.

`z_step == 0` happens because `generate_lens_plane_redshifts` is:

```python
def generate_lens_plane_redshifts(zlens, zsource):
    zmin = lenscone_default.default_zstart   # 0.01
    zstep = lenscone_default.default_z_step  # 0.02
    if zlens is None:
        redshifts = np.arange(zmin, zsource, zstep)
    else:
        front_z = np.arange(zmin, zlens, zstep)
        back_z = np.arange(zlens, zsource, zstep)
        redshifts = np.append(front_z, back_z)
    delta_zs = []
    for i in range(0, len(redshifts) - 1):
        delta_zs.append(redshifts[i + 1] - redshifts[i])
    delta_zs.append(zsource - redshifts[-1])
    return list(np.round(redshifts, 2)), np.round(delta_zs, 2)
```

`np.arange` over floating-point step `0.02` is numerically fragile (0.02 is not
exactly representable in IEEE-754; the actual step is
≈ `0.020000000000000004`). At certain `zlens` values, the accumulated FP error
pushes the last `front_z` element across the exclusive upper bound, so the
last element of `front_z` rounds (to 2 dp) to the same value as the first
element of `back_z`, which equals `zlens`. After `np.round(redshifts, 2)`, the
plane list contains `zlens` twice in a row, the consecutive difference is
`0.0`, the host-plane index lands on one of the duplicates, and
`delta_z = 0` flows into the two-halo term.

## Minimal reproducer

No mejiro / external data required:

```python
import numpy as np
from pyHalo.utilities import generate_lens_plane_redshifts

# Affected pair from a Roman-style population (any of these reproduce the bug):
zlens, zsource = 2.23, 3.21
redshifts, deltas = generate_lens_plane_redshifts(zlens, zsource)

idx = int(np.argmin(np.abs(np.array(redshifts) - zlens)))
print("planes near host:", redshifts[idx-1:idx+2])
print("delta_z at host plane:", deltas[idx])
print("duplicate count at zlens:", redshifts.count(zlens))
# planes near host: [2.21, 2.23, 2.23]
# delta_z at host plane: 0.0
# duplicate count at zlens: 2
```

End-to-end reproducer that actually triggers the colossus exception:

```python
import numpy as np
from pyHalo.preset_models import preset_model_from_name
CDM = preset_model_from_name('CDM')
CDM(z_lens=2.23, z_source=3.21, log_m_host=13.3, cone_opening_angle_arcsec=5.0)
# Exception: R = 0.00e+00 is too small (min. R = 1.00e-03)
```

## Range of affected `(zlens)` values

Brute-force sweep of `zlens ∈ {0.01, 0.02, …, 5.00}` paired with a range of
`zsource > zlens`: the duplicate-plane condition fires at the following
`zlens` values (all 2-decimal-place):

* 2.23, 2.25, 2.27, 2.29, 2.31, 2.33, 2.35, 2.37, 2.39, 2.41, 2.43, 2.45, 2.47, 2.49
* 2.91, 2.93, 2.95, 2.97, 2.99
* 3.41, 3.43, 3.45, 3.47, 3.49
* 3.93, 3.95, 3.97, 3.99
* 4.03, 4.07, …

i.e. clusters of odd-hundredth `zlens` values where the FP accumulation in
`np.arange(0.01, zlens, 0.02)` ticks over the exclusive bound.

Empirically (against a Roman lens-population survey of ~100k systems): every
failure has `round(z_lens, 2)` on an odd hundredth in this set; the `zsource`
distribution of failures is otherwise unremarkable.

## Suggested fix

Two equally valid options:

1. **Build planes deterministically with `np.linspace`** rather than
   `np.arange`. For example:

   ```python
   def _planes(zlo, zhi, zstep):
       if zhi <= zlo:
           return np.array([])
       n = int(np.round((zhi - zlo) / zstep))
       # values in [zlo, zhi) at spacing zstep, using integer-count arithmetic
       return zlo + zstep * np.arange(n)
   ```

   This avoids accumulating FP error and produces consistent counts.

2. **Deduplicate after rounding.** Cheaper, more local:

   ```python
   rounded = np.round(redshifts, 2)
   # drop consecutive duplicates
   keep = np.concatenate(([True], np.diff(rounded) > 0))
   redshifts = rounded[keep].tolist()
   ```

   then recompute `delta_zs` from the deduped list.

Option 1 is cleaner; option 2 is a one-line patch.

A regression test should assert:

```python
for zlens in [2.23, 2.31, 2.41, 2.47, 2.95, 3.49, 3.99]:
    r, d = generate_lens_plane_redshifts(zlens, zlens + 1.0)
    assert len(r) == len(set(r)), f"duplicate planes at zlens={zlens}: {r}"
    assert np.all(np.asarray(d) > 0), f"zero delta_z at zlens={zlens}"
```

## Workaround currently used in mejiro

`mejiro/pipeline/_03_rung_1.py` and `mejiro/pipeline/_03_generate_subhalos.py`
retry once with `z_lens` nudged by +0.01 (~10 Mpc at z ≈ 2.4, well within the
2-dp rounding mejiro already applies). The shifted `z_lens` lands on an
even-hundredth value and avoids the duplicate-plane condition. This recovers
the ~0.1% of systems that would otherwise be lost.

## Claude Code prompt for opening an upstream PR

Paste the block below into Claude Code from a clone of
`https://github.com/dangilman/pyHalo`.

````
The function `pyHalo.utilities.generate_lens_plane_redshifts` produces
duplicate redshift planes at the host (`zlens`) plane for certain values of
`zlens`. This causes `pyHalo/Rendering/two_halo.py::two_halo_enhancement_factor`
to compute `delta_z = 0` for the host plane, which makes `rmax = rmin = 0`
in the `scipy.integrate.quad(_boost_integrand, rmin, rmax, ...)` call, which
in turn calls colossus's `Cosmology.correlationFunction` at `R = 0` and
raises:

    Exception: R = 0.00e+00 is too small (min. R = 1.00e-03)

Reproducer (no external data):

    import numpy as np
    from pyHalo.utilities import generate_lens_plane_redshifts
    r, d = generate_lens_plane_redshifts(2.23, 3.21)
    idx = int(np.argmin(np.abs(np.array(r) - 2.23)))
    assert r.count(2.23) == 2          # bug: duplicate plane
    assert d[idx] == 0.0               # bug: zero spacing
    # And end-to-end:
    from pyHalo.preset_models import preset_model_from_name
    CDM = preset_model_from_name('CDM')
    CDM(z_lens=2.23, z_source=3.21, log_m_host=13.3,
        cone_opening_angle_arcsec=5.0)
    # raises: Exception: R = 0.00e+00 is too small (min. R = 1.00e-03)

Root cause: `np.arange(0.01, zlens, 0.02)` followed by
`np.arange(zlens, zsource, 0.02)` is FP-fragile (0.02 is not exactly
representable; the effective step is ~0.020000000000000004). At certain
`zlens` values, the accumulated error pushes the last `front_z` element
across the exclusive upper bound, so after `np.round(redshifts, 2)` the
plane at `zlens` appears twice.

Affected `zlens` values (2-dp, sweep over [0.01, 5.00]):
2.23-2.49 odd, 2.91-2.99 odd, 3.41-3.49 odd, 3.93-3.99 odd, 4.03, 4.07, …

Please:

1. Fix `generate_lens_plane_redshifts` in `pyHalo/utilities.py` so it never
   returns duplicated consecutive redshifts. Either (a) build the front/back
   plane lists with integer-count arithmetic
   (e.g. `zlo + zstep * np.arange(n)` where `n = round((zhi - zlo)/zstep)`),
   or (b) drop consecutive duplicates after the existing `np.round`. Update
   `delta_zs` accordingly. Apply the same fix to the other call site in
   `pyHalo/utilities.py` near line 188 and to
   `pyHalo/Cosmology/cosmology.py:123,134` if they share the same pattern.

2. Add a regression test (pytest) that loops over the affected `zlens`
   values above and asserts:
     - no duplicates in the returned redshifts list
     - every entry of `delta_zs` is > 0
   and a second test that calls `CDM(z_lens=2.23, z_source=3.21,
   log_m_host=13.3, cone_opening_angle_arcsec=5.0)` and asserts it returns
   a non-empty realization (currently raises).

3. Open a PR with a short summary of the FP root cause and the fix.
````
