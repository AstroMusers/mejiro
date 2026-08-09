# lenstronomy: `np.fft.rfftn` called with `s=` but `axes=None`

## Symptom

mejiro's test suite emits **169** copies of this NumPy 2.0 deprecation, split across three lines of
one file:

```
/…/lenstronomy/ImSim/Numerics/convolution.py:121: DeprecationWarning: `axes` should not be
`None` if `s` is not `None` (Deprecated in NumPy 2.0). In a future version of NumPy, this
will raise an error and `s[i]` will correspond to the size along the transformed axis
specified by `axes[i]`. To retain current behaviour, pass a sequence [0, ..., k-1] to
`axes` for an array of dimension k.
    sp1 = np.fft.rfftn(in1, fshape)
```

| Line | Call | Warnings in mejiro's suite |
|-----:|------|---------------------------:|
| 121 | `sp1 = np.fft.rfftn(in1, fshape)` | 70 |
| 122 | `ret = np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy()` | 70 |
| 174 | `sp2 = np.fft.rfftn(in2, fshape)` | 29 |

Line numbers are **identical in 1.13.3 and 1.14.2** (checked against both sdists), so this is
current on `main` as of lenstronomy 1.14.2.

NumPy states it "will raise an error" in a future version, so this is a future breakage, not just
noise.

## Root cause

`PixelKernelConvolution._static_fft` and `_static_pre_compute` (the `convolution_type="fft_static"`
path, which is lenstronomy's **default** — see `PixelKernelConvolution.__init__`) pass `fshape`
positionally, which binds it to `s`:

```python
fshape = [fftpack.next_fast_len(int(d)) for d in shape]   # one entry per dimension
...
sp1 = np.fft.rfftn(in1, fshape)                            # s=fshape, axes=None
```

NumPy 2.0 deprecated the `s`-without-`axes` combination because the pairing between `s[i]` and the
transformed axis is about to change.

Since `shape = s1 + s2 - 1` is elementwise over the input dimensions, `len(fshape) == in1.ndim`
always holds here. The current behaviour is therefore exactly "transform axes `0..ndim-1`", which is
what NumPy tells you to make explicit.

## Minimal reproducer

No mejiro or external data required:

```python
import warnings
import numpy as np
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution

kernel = np.ones((5, 5)); kernel /= kernel.sum()
image = np.random.rand(31, 31)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    conv = PixelKernelConvolution(kernel, convolution_type='fft_static')
    conv.convolution2d(image)

hits = [x for x in w if 'axes' in str(x.message)]
print(f'{len(hits)} axes DeprecationWarnings')
for x in hits:
    print('  line', x.lineno)
# 3 axes DeprecationWarnings
#   line 174
#   line 121
#   line 122
```

## Suggested fix

### Option 1 — pass `axes` explicitly (low risk, recommended)

```python
# convolution.py, _static_fft, lines 121-122
axes = tuple(range(in1.ndim))
sp1 = np.fft.rfftn(in1, fshape, axes=axes)
ret = np.fft.irfftn(sp1 * sp2, fshape, axes=axes)[fslice].copy()

# convolution.py, _static_pre_compute, line 174
sp2 = np.fft.rfftn(in2, fshape, axes=tuple(range(in2.ndim)))
```

Verified **bitwise identical** to the current output, and warning-free:

```python
import warnings, numpy as np
rng = np.random.default_rng(0)
in1, fshape = rng.random((37, 41)), [48, 50]

old = np.fft.rfftn(in1, fshape)
new = np.fft.rfftn(in1, fshape, axes=tuple(range(in1.ndim)))
assert np.array_equal(old, new)     # passes
# same for irfftn with axes=(0, 1)
```

### Option 2 — drop the vendored copy (larger, better)

`_static_fft` is a vendored copy of an old `scipy.signal.fftconvolve`, including this:

```python
# Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
# sure we only call rfftn/irfftn from one thread at a time.
if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
```

NumPy's FFT routines have been thread-safe since NumPy 1.9 (2014), so `_rfft_mt_safe` is always
true and the lock and its entire `else` fallback branch are dead code. The only thing the vendored
copy still buys over `scipy.signal.fftconvolve` is caching the transformed kernel (`sp2`) across
calls — which is the actual point of `fft_static` and is worth keeping.

A middle path: keep the caching, delete the `_rfft_lock` / `_rfft_mt_safe` machinery and the
complex-FFT fallback, and apply Option 1 to what remains. That removes ~25 lines of dead code along
with the deprecation.

Option 1 alone is the safe PR; mention Option 2 and let the maintainers decide.

## Claude Code prompt for opening an upstream PR

Paste the block below into Claude Code from a clone of
`https://github.com/lenstronomy/lenstronomy`.

````
`lenstronomy/ImSim/Numerics/convolution.py` calls `np.fft.rfftn` / `np.fft.irfftn`
with a shape argument but no `axes`, which NumPy 2.0 deprecated and says will
become an error:

    DeprecationWarning: `axes` should not be `None` if `s` is not `None`
    (Deprecated in NumPy 2.0). In a future version of NumPy, this will raise an
    error and `s[i]` will correspond to the size along the transformed axis
    specified by `axes[i]`.

Three call sites, in the `convolution_type="fft_static"` path, which is the
default for PixelKernelConvolution:

    line 121  sp1 = np.fft.rfftn(in1, fshape)                        (_static_fft)
    line 122  ret = np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy()  (_static_fft)
    line 174  sp2 = np.fft.rfftn(in2, fshape)                        (_static_pre_compute)

Reproducer (no external data):

    import warnings, numpy as np
    from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
    kernel = np.ones((5, 5)); kernel /= kernel.sum()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        PixelKernelConvolution(kernel, convolution_type='fft_static'
            ).convolution2d(np.random.rand(31, 31))
    assert len([x for x in w if 'axes' in str(x.message)]) == 3

Please fix by passing `axes` explicitly. `fshape` is built as
`[fftpack.next_fast_len(int(d)) for d in shape]` where `shape = s1 + s2 - 1`, so
it always has exactly one entry per input dimension, and `tuple(range(ndim))`
reproduces the current behaviour exactly:

    axes = tuple(range(in1.ndim))
    sp1 = np.fft.rfftn(in1, fshape, axes=axes)
    ret = np.fft.irfftn(sp1 * sp2, fshape, axes=axes)[fslice].copy()
    ...
    sp2 = np.fft.rfftn(in2, fshape, axes=tuple(range(in2.ndim)))

I verified this is bitwise identical to the current output:

    old = np.fft.rfftn(in1, fshape)
    new = np.fft.rfftn(in1, fshape, axes=tuple(range(in1.ndim)))
    assert np.array_equal(old, new)

Separately, and only if the maintainers want it in the same PR: `_static_fft` is a
vendored copy of an old scipy.signal.fftconvolve and still carries a `_rfft_lock` /
`_rfft_mt_safe` guard for "pre-1.9 NumPy FFT routines are not threadsafe". NumPy's
FFT has been threadsafe since 1.9 (2014), so that lock and its complex-FFT fallback
branch are dead code. Worth flagging in the PR description rather than doing
unasked.

Run the existing convolution tests and confirm outputs are unchanged.
````
