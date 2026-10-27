# Sub-pixel phases in the Roman image pipeline

A source center can fall between detector-pixel centers. Its sub-pixel phase is the fractional displacement from the nearest pixel center, measured in detector-pixel units. A phase of zero places the source at the center of a pixel.

## L2 images

The L2 path has no dither WCS from which to measure a phase. The `subpixel_phase(seed, lens_name)` function assigns each system a deterministic pair of offsets in `[-0.5, 0.5)`. The same seed and lens name give the same phase across bands and regardless of processing order. Tests check repeatability, range, and approximate uniformity across 10,000 generated names. These generated offsets make L2 runs reproducible. They do not measure Roman pointing offsets.

## L3 mosaics

The L3 path derives each exposure's phase from its dither WCS. The `bin_to_native` function shifts an oversampled step-04 image to that phase, then sums each oversampled block into a detector pixel. It rejects nonzero shifts at oversample 1 because a detector-resolution image cannot recover the phase-dependent sampling of an undersampled source.

For the implementation details, see [`_05_romanisim.py`](../mejiro/pipeline/_05_romanisim.py), the [sub-pixel phase tests](../tests/test_pipeline/test_subpixel_phase.py), and the [L3 dither registration note](l3_dither_registration.md).
