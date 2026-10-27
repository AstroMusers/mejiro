# lenstronomy `np.fft.rfftn` call with `s=` and `axes=None`

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

NumPy says this call pattern will become an error in a future release.

## Root cause

`PixelKernelConvolution._static_fft` and `_static_pre_compute` (the `convolution_type="fft_static"`
path, which is lenstronomy's **default** – see `PixelKernelConvolution.__init__`) pass `fshape`
positionally, which binds it to `s`:

```python
fshape = [fftpack.next_fast_len(int(d)) for d in shape]   # one entry per dimension
...
sp1 = np.fft.rfftn(in1, fshape)                            # s=fshape, axes=None
```

NumPy 2.0 deprecated the `s`-without-`axes` combination because the pairing between `s[i]` and the
transformed axis is about to change.

Since `shape = s1 + s2 - 1` is elementwise over the input dimensions, `len(fshape) == in1.ndim`
always holds here. The current behavior is therefore exactly "transform axes `0..ndim-1`", which is
what NumPy tells you to make explicit.

## Minimal reproducer

Reproduce the warning without Mejiro or external data.

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

### Option 1 – pass `axes` explicitly (low risk, recommended)

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

### Option 2 – replace the vendored implementation

`_static_fft` is a vendored copy of an old `scipy.signal.fftconvolve`, including this:

```python
# Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
# sure we only call rfftn/irfftn from one thread at a time.
if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
```

NumPy's FFT routines have been thread-safe since NumPy 1.9 (2014), so the `_rfft_mt_safe` guard and
its lock fallback are unnecessary. The vendored implementation's remaining advantage over
`scipy.signal.fftconvolve` is caching the transformed kernel (`sp2`) across calls. This cache avoids
recomputing the kernel transform on every call.

Keeping the cache while removing `_rfft_lock`, `_rfft_mt_safe`, and the complex-FFT fallback would
remove about 25 lines of obsolete code alongside the deprecation fix.
