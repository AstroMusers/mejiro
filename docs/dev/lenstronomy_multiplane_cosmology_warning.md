# lenstronomy multi-plane "Cosmology is provided" warning is unconditional

## Symptom

Every multi-plane `LensModel` construction emits:

```
/…/lenstronomy/LensModel/MultiPlane/multi_plane.py:81: UserWarning: Cosmology is
provided. Make sure your cosmological model is consistent with the cosmology_model
argument.
```

It fires whether or not the caller passed a cosmology, and whether or not the caller ever touched
`cosmology_model`. In mejiro's test suite it appears 3 times, from
[mejiro/strong_lens.py:200](../../mejiro/strong_lens.py#L200) `get_realization_kappa`.

Present and unchanged in **both** lenstronomy 1.13.3 and 1.14.2 (checked against both sdists).

## Root cause

`MultiPlane.__init__` (`lenstronomy/LensModel/MultiPlane/multi_plane.py:76-83`) branches on `cosmo`:

```python
if cosmo is None and cosmology_model == "FlatLambdaCDM":
    cosmo = default_cosmology.get()
elif cosmo is None and cosmology_model != "FlatLambdaCDM":
    cosmo = get_astropy_cosmology(cosmology_model=cosmology_model)
else:
    warnings.warn(
        "Cosmology is provided. Make sure your cosmological model is consistent with the cosmology_model argument."
    )
```

Both guarded branches require `cosmo is None`, so the `else` catches **every** non-None `cosmo`,
including the ordinary case of a caller supplying a cosmology while leaving `cosmology_model` at its
default `"FlatLambdaCDM"`. In that case the warning asks the user to verify consistency with an
argument they never set.

It is worse than that in practice, because `MultiPlane` is rarely constructed directly.
`LensModel.__init__` (`lenstronomy/LensModel/lens_model.py:98-101`) resolves the default *first*:

```python
if cosmo is None and cosmology_model == "FlatLambdaCDM":
    cosmo = default_cosmology.get()          # cosmo is now non-None
...
                cosmo=cosmo,                 # line 194: handed to MultiPlane
```

So `MultiPlane` **never** receives `cosmo=None` through the public API, the first two branches are
unreachable from `LensModel`, and the warning is unconditional: it fires on every single multi-plane
model, with no way for a caller to avoid it.

Note that `lens_model.py:102-106` already implements the intended logic correctly one layer up — it
warns only when `cosmology_sampling is True`. `MultiPlane`'s copy is the unguarded one.

## Minimal reproducer

No external data. Both of these warn, including the second, which passes no cosmology at all:

```python
import warnings
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel

def count(**kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        LensModel(lens_model_list=['SIS'], z_lens=0.5, z_source=1.5,
                  lens_redshift_list=[0.5], multi_plane=True, **kwargs)
    return len([x for x in w if 'Cosmology is provided' in str(x.message)])

print('explicit cosmo, default cosmology_model:', count(cosmo=FlatLambdaCDM(H0=70, Om0=0.3)))
print('no cosmo at all:                        ', count())
# explicit cosmo, default cosmology_model: 1
# no cosmo at all:                         1
```

The second line is the bug in its clearest form: a user who passed nothing is told that "Cosmology
is provided."

## Suggested fix

Warn only when the caller supplies **both** a `cosmo` and a non-default `cosmology_model` — the one
situation where the two can actually disagree:

```python
if cosmo is None:
    if cosmology_model == "FlatLambdaCDM":
        cosmo = default_cosmology.get()
    else:
        cosmo = get_astropy_cosmology(cosmology_model=cosmology_model)
elif cosmology_model != "FlatLambdaCDM":
    warnings.warn(
        "Both cosmo and a non-default cosmology_model were provided; make sure they are "
        "consistent. cosmo takes precedence."
    )
```

This keeps the warning for the genuinely ambiguous case and silences it everywhere else. It also
makes the `cosmo is None` branches reachable again if `MultiPlane` is constructed directly.

Regression test:

```python
def test_multiplane_does_not_warn_for_ordinary_construction():
    import warnings
    from astropy.cosmology import FlatLambdaCDM
    from lenstronomy.LensModel.lens_model import LensModel
    for kwargs in ({}, {'cosmo': FlatLambdaCDM(H0=70, Om0=0.3)}):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            LensModel(lens_model_list=['SIS'], z_lens=0.5, z_source=1.5,
                      lens_redshift_list=[0.5], multi_plane=True, **kwargs)
```

## Current Mejiro behavior

[mejiro/strong_lens.py:200](../../mejiro/strong_lens.py#L200) passes
`cosmo=self.cosmo` to every `multi_plane=True` `LensModel`. Mejiro needs its own
cosmology rather than Astropy's process-wide default. The unconditional upstream
warning therefore appears during this normal construction.
