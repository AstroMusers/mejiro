# pyHalo: unfinished `np.trapz` → `np.trapezoid` migration

## Symptom

Running mejiro's test suite against `pyhalo==1.4.3` produces **4,838** copies of:

```
/…/pyHalo/Halos/HaloModels/NFW_core_trunc.py:174: DeprecationWarning: `trapz` is
deprecated. Use `trapezoid` instead, or one of the numerical integration functions
in `scipy.integrate`.
    mass_3d = np.trapz(4 * np.pi * r ** 2 * rho, r)
```

That is 95% of every warning the suite emits. `np.trapz` was deprecated in NumPy 2.0 and **removed
in NumPy 2.4**, so this is not cosmetic — it is why
[pyproject.toml](../../pyproject.toml) pins `numpy<2.4.0`:

```toml
"numpy<2.4.0",  # including this for now because pyhalo==1.4.3 uses np.trapz which is removed in 2.4.0
```

## Status upstream

**The specific line mejiro trips is already fixed.** In pyhalo 1.4.9, `NFW_core_trunc.py:174` reads
`mass_3d = np.trapezoid(...)`. Bumping the pin would clear all 4,838 warnings.

But the migration is **half-finished**. Two live `np.trapz` calls remain in 1.4.9:

| File | Line | Call |
|------|-----:|------|
| `pyHalo/Halos/HaloModels/TNFW.py` | 96 | `return np.trapz(kappa * 2 * np.pi * x, x) * sigma_crit_arcsec` |
| `pyHalo/Halos/HaloModels/powerlaw.py` | 177 | `m = np.trapz(4 * np.pi * r**2 * density, r)` |

(There are also commented-out `np.trapz` occurrences at `NFW_core_trunc.py:503,505` and
`Rendering/MassFunctions/mass_function_base.py:256,257` — those are dead code, not part of the fix.
Be careful grepping: a naive `grep np.trapz` makes 1.4.9 look unfixed.)

This leaves pyHalo 1.4.9 in a contradictory state:

- `np.trapezoid` **requires numpy ≥ 2.0** (it does not exist in 1.x)
- `np.trapz` **requires numpy < 2.4** (removed in 2.4)

So pyHalo 1.4.9 silently only works on `2.0 <= numpy < 2.4`, and it declares neither bound. Its
`requires_dist` is just `['Click>=6.0']` — no numpy dependency at all.

## Minimal reproducer

Against any pyHalo version, with numpy ≥ 2.0:

```python
import warnings
import numpy as np
from pyHalo.preset_models import preset_model_from_name

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    CDM = preset_model_from_name('CDM')
    CDM(z_lens=0.5, z_source=1.5, log_m_host=13.3, cone_opening_angle_arcsec=5.0)

trapz = [x for x in w if 'trapz' in str(x.message)]
print(f'{len(trapz)} trapz DeprecationWarnings')
for x in trapz[:1]:
    print(x.filename, x.lineno)
```

And the removal, which is the part that actually breaks:

```python
import numpy as np
assert not hasattr(np, 'trapz'), 'numpy < 2.4 — upgrade to see the real failure'
# On numpy >= 2.4 the calls above raise AttributeError instead of warning.
```

## Suggested fix

Mechanical, three parts:

1. Replace the two remaining calls with `np.trapezoid`, matching what `NFW_core_trunc.py:174`
   already does:

   ```python
   # pyHalo/Halos/HaloModels/TNFW.py:96
   return np.trapezoid(kappa * 2 * np.pi * x, x) * sigma_crit_arcsec

   # pyHalo/Halos/HaloModels/powerlaw.py:177
   m = np.trapezoid(4 * np.pi * r**2 * density, r)
   ```

   `np.trapezoid` is a pure rename — same signature, same algorithm, same result. No numerical
   change.

2. Declare the numpy floor that `np.trapezoid` already imposes. In `setup.py` / `pyproject.toml`:

   ```
   numpy>=2.0
   ```

   Without this, `pip install pyhalo` on a numpy 1.x environment resolves happily and then fails at
   runtime inside `NFW_core_trunc.profile_args`.

3. If pyHalo wants to keep supporting numpy 1.x, use a module-level alias instead of (1) and (2):

   ```python
   # pyHalo/Halos/HaloModels/_compat.py
   import numpy as np
   trapezoid = getattr(np, 'trapezoid', None) or np.trapz
   ```

   and import `trapezoid` at the four live call sites. This is the only option that works across
   numpy 1.x, 2.0–2.3, **and** 2.4+.

A regression test that would have caught this:

```python
import warnings, numpy as np, pyHalo, pkgutil, importlib, inspect, re

def test_no_deprecated_trapz():
    offenders = []
    for mod in pkgutil.walk_packages(pyHalo.__path__, 'pyHalo.'):
        try:
            src = inspect.getsource(importlib.import_module(mod.name))
        except Exception:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if re.search(r'^\s*[^#]*\bnp\.trapz\b', line):
                offenders.append(f'{mod.name}:{i}')
    assert not offenders, f'np.trapz removed in numpy 2.4: {offenders}'
```

## What mejiro does in the meantime

Nothing. mejiro stays on `pyhalo==1.4.3` and accepts the 4,838 warnings.

Bumping to 1.4.9 is a separate, science-affecting change that needs its own before/after validation:
it crosses six releases, and `NFW_core_trunc.profile_args` also changes the mass-conservation
integration grid from `np.logspace(-4, log10(r_match/rs), 1000)` to the same range with **250**
points. That is a real numerical difference in the `alpha_Rs` normalisation, not just a rename.

Note also that bumping would **not** lift the `numpy<2.4.0` pin, because of the two calls above.
The pin can only be lifted once this upstream fix lands.

One thing the bump does *not* change: `pyHalo.utilities.generate_lens_plane_redshifts` is identical
in 1.4.3 and 1.4.9, so the duplicate-lens-plane workaround described in
[pyhalo_lens_plane_bug.md](pyhalo_lens_plane_bug.md) stays necessary either way.

## Claude Code prompt for opening an upstream PR

Paste the block below into Claude Code from a clone of `https://github.com/dangilman/pyHalo`.

````
pyHalo's migration from the deprecated `np.trapz` to `np.trapezoid` is incomplete,
which leaves the package unable to run on NumPy 2.4+ (where `np.trapz` was removed)
while simultaneously requiring NumPy 2.0+ (where `np.trapezoid` was introduced).

Already migrated:
    pyHalo/Halos/HaloModels/NFW_core_trunc.py:174   np.trapezoid(...)

Still using the removed function:
    pyHalo/Halos/HaloModels/TNFW.py:96
        return np.trapz(kappa * 2 * np.pi * x, x) * sigma_crit_arcsec
    pyHalo/Halos/HaloModels/powerlaw.py:177
        m = np.trapz(4 * np.pi * r**2 * density, r)

(Ignore the commented-out occurrences at NFW_core_trunc.py:503,505 and
Rendering/MassFunctions/mass_function_base.py:256,257 — dead code.)

Consequence: on NumPy >= 2.4 these two call sites raise
`AttributeError: module 'numpy' has no attribute 'trapz'`. Downstream packages are
currently pinning `numpy<2.4.0` to work around it.

Please:

1. Replace both calls with `np.trapezoid`. This is a pure rename — identical
   signature and result, no numerical change.

2. Declare the NumPy floor that `np.trapezoid` already imposes. pyHalo's install
   requirements are currently just `Click>=6.0` and specify no numpy bound at all,
   so `pip install pyhalo` succeeds on numpy 1.x and then fails at runtime inside
   `NFW_core_trunc.profile_args`. Add `numpy>=2.0`.

   Alternatively, if pyHalo wants to keep numpy 1.x support, add a small compat
   shim instead and use it at all live call sites:

       trapezoid = getattr(np, 'trapezoid', None) or np.trapz

   That is the only approach that works on numpy 1.x, 2.0-2.3, and 2.4+.

3. Add a regression test that greps the package for `np.trapz` outside comments, so
   the remaining call sites cannot reappear.

Verify with `pytest` and, if you can, by installing numpy>=2.4 in a scratch env and
constructing a TNFW and a powerlaw halo profile.
````
