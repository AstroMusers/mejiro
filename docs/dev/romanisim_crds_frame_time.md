# romanisim with CRDS: frame time and ePSF lookup

Status: open upstream (romanisim 0.14.0); worked around in `mejiro/pipeline/_05_romanisim.py` (`_use_crds_reference_data`) and in `mejiro-projects/lens_search/lens_search_hlwas.ipynb`.
Last verified: 2026-11-15, against romanisim 0.14.0 and CRDS context `roman_0078.pmap`.

## The ramp fit ignores the CRDS frame time

With `usecrds=True`, romanisim takes the exposure time it accumulates counts over from the CRDS
MA-table reference: `frame_time` times the last read (`simulate_counts`, and
`util.add_more_metadata`). Its read times and ramp fit, though, use the module constant
`romanisim.models.parameters.read_time` (`l1.read_pattern_to_tij`, `ramp.read_pattern_to_tbar`).
For MA table 17 in `roman_0078.pmap` the CRDS frame time is 5.14116 s and `read_time` is
3.16247 s, so every slope in the L2 (sky, dark, sources, and mejiro's lenses) comes out 1.63
times too high.

`_05_romanisim` sets `parameters.read_time` to the MA table's frame time in each worker when
`exposure.usecrds` is on. `test_process_batch_l2_with_crds` in
`tests/test_pipeline/test_romanisim_crds.py` (run with `--run-crds`) failed by exactly that
factor before the fix. `romanisim.image.inject_sources_into_l2` has the same assumption (there's
a comment about it at the exposure-time line), so the notebook's injection helper sets the read
time to each L2's own `meta.exposure.frame_time` while it works.

## The ePSF lookup names the detector `SCAxx`

`romanisim.psf.get_epsf_from_crds` asks CRDS for detector `SCAxx`, but the CRDS rules name
detectors `WFIxx`, so `psftype='epsf'` fails with a `CrdsLookupError`. `make_psf(variable=True)`
also doesn't pass the date on, so the lookup falls back to a default date. The notebook swaps in
a corrected copy when `PSF_TYPE = 'epsf'`. mejiro's pipeline isn't affected: step 05 passes an
empty catalog with `psftype='galsim'`, since step 04 already applied the PSF.

When romanisim fixes either, drop the workaround and this note.
