# Oversampled step-04 rendering

The `synthetic_image.oversample` setting controls the sampling scale of step-04 images. For an oversampling factor $N > 1$, step 04 renders at `pixel_scale / N` and leaves the detector-pixel integral unapplied. Step 05 then shifts the oversampled image to the exposure's sub-pixel phase and sums each $N \times N$ block into detector pixels.

The PSF kernel must match the grid that it will convolve. At `oversample > 1`, step 04 requests the STPSF kernel at the configured oversampling factor and uses it without degrading it. Degrading that kernel would apply the detector-pixel response a second time. The matching oversampled kernel must be present in the PSF cache.

`supersampling_factor` is a separate setting. When `oversample > 1`, it controls sampling within each sub-pixel rather than within a detector pixel. Choose it on that basis. The [supersampling convergence script](../projects/roman_data_challenge/supersampling_convergence.py) compares renders after binning them to detector resolution.

With `oversample = 1`, the image is already at detector resolution and step 04 follows the kernel-degradation path. Romanisim cannot apply a nonzero sub-pixel shift to this image without shifting at native resolution. For the phase-aware oversampled path, see the [L3 dither registration note](l3_dither_registration.md) and [sub-pixel phase guide](subpixel_phase.md).

Implementation details are in [`_04_create_synthetic_images.py`](../mejiro/pipeline/_04_create_synthetic_images.py) and [`_05_romanisim.py`](../mejiro/pipeline/_05_romanisim.py).
