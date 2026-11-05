## Unreleased (changes since v3.0.0)

Add to this file as changes land on `main`; rename it to `vX.Y.Z.md` at the next release. Under
semantic versioning, the breaking changes below mean the next release should be a major version.

### Upgrading from v3.0.0

Configs written for v3.0.0 need these changes (see the
[configuration reference](https://mejiro.readthedocs.io/en/latest/usage/configuration.html)):

- Add `subhalos.serialization`, `synthetic_image.serialization` and `imaging.serialization`
  (each `full` or `lightweight`). All three are required.
- Add `dataset.labeled` (`True` to write truth values into the HDF5 file, `False` for a
  separate answer-key CSV). Step 06 requires it.
- Rename `cores.script_05` to `cores.script_05_galsim`. `cores.script_01` is no longer used.

Scripts and output directories that changed:

- `_01_run_survey_simulation.py` is gone; use `_01a_generate_galaxy_tables.py` then
  `_01b_run_survey_simulation.py`.
- The step 04 variants (`_04_create_synthetic_images_interpol.py`,
  `_04_jax_create_synthetic_images.py`) are merged into `_04_create_synthetic_images.py`. JAX
  ray-shooting is selected with `jaxtronomy.use_jax`.
- `_05_create_exposures.py` is now `_05_galsim.py` (output directory `05_galsim/`).
- The romanisim scripts are merged into one step, `_05_romanisim.py` (output directory
  `05_romanisim/`), with `--level l2|l3` and `--dither-pattern`.
- `_06_h5_export_romanisim.py` is merged into `_06_h5_export.py`; `_07_subhalo_export.py` is
  removed.

### Breaking changes

- **Python 3.12 or later is required** (v3.0.0's `pyproject.toml` allowed 3.11). 3.12 and 3.13
  are supported; CI tests 3.13.
- **The `romanisim` `Exposure` engine is removed**; `Exposure(engine='romanisim')` raises
  `NotImplementedError`. Roman L2/L3 simulation runs as pipeline step `_05_romanisim`.
- **The HST and LSST instrument classes moved off `main`** to the `feature/hst` and
  `feature/lsst` branches while they're unfinished. Populations drawn with
  `pipeline_label: all` no longer include HST (WFC3/UVIS) magnitudes.
- **The Roman Data Challenge configs** (rung 0/1, labeled and unlabeled) are no longer shipped
  in `mejiro/data/mejiro_config/`.
- **`all.yaml`, `hwo.yaml` and `jwst.yaml` are no longer shipped** in
  `mejiro/data/mejiro_config/`, and neither are their SkyPy configs in `mejiro/data/skypy/`
  (`all`, `hwo`, `jwst`, `slhammocks_all`, `slhammocks_hwo` and `slhammocks_jwst`). The
  SkyPy configs no shipped config used are gone too: `euclid-lsst-roman-like`, `hwo_dev`,
  `roman_deep`, `roman_dev`, `roman_medium`, `roman_wide` and `roman.yml`. A config that names
  a removed SkyPy config now stops at step 01a with `FileNotFoundError`.
- The config, script and output-directory changes listed under "Upgrading from v3.0.0".

### Pipeline

- **Lightweight serialization** to cut disk use: step 03 can drop the pyHalo realization
  object (`subhalos.serialization`), step 04 can write `.npz` synthetic images
  (`synthetic_image.serialization`), and step 05 (GalSim) can write `.npz` exposures
  (`imaging.serialization`). `mejiro.utils.util.load_synthetic_image` and `load_exposure`
  read either format. `compact_03_realizations.py` compacts existing step 03 output.
- **JAX ray-shooting** with jaxtronomy in steps 01b, 03, 04 and `calculate_snrs`, with CPU
  and GPU execution paths (`jaxtronomy.use_jax`, `jaxtronomy.jax_platform`).
- **Roman L2/L3 simulation** in `_05_romanisim`: dithered pointings with the official WFI
  dither patterns (`mejiro.point_wfi`), deterministic sub-pixel phases, and L3 mosaics built
  with romancal.
- **Oversampled rendering** in step 04 (`synthetic_image.oversample`), which `_05_romanisim`
  bins to detector pixels at each dither's sub-pixel phase.
- **Deflector-only systems**: render a fraction of systems with lens light only
  (`synthetic_image.deflector_only_fraction`), recorded in the HDF5 file or answer key.
- Every step script accepts `--data_dir`.
- `logging_level` now controls all pipeline console output; Python warnings go through
  `logging`.
- Ctrl+C stops worker processes.

### Science and analysis

- Lens subtraction: optional sky pedestal, Poisson weighting, and a two-component
  bulge + disk Sersic fit.
- NFW halo properties are read from slsim's `mass_properties`.
- pyHalo realizations are retried with a nudged lens redshift when pyHalo fails on duplicate
  lens-plane redshifts.
- The Roman PSFs shipped in `mejiro/data/psfs/roman/` were regenerated with STPSF 2.3.0. They
  differ from the old ones by 2-3% of the peak in most bands, 7% in F087 and 12% in F062, so
  results that use them can change.

### Bug fixes

- L3 co-adds: negative drizzle weights fixed, and the recorded depth is the true co-add
  depth; L3 cutout orientation fixed (per-SCA rotation and parity flip); dither tiles
  rendered at their true sub-pixel position.
- `_05_romanisim` raises when every batch fails, instead of continuing silently.
- `_05_galsim` skips corrupted `SyntheticImage` pickles instead of crashing.
- Galaxy-table seeding is deterministic across processes, and step 04's detector-position
  draw is seeded.
- `simple.yaml` and `roman_test.yaml` set `dataset.labeled`, so step 06 no longer stops with
  `KeyError: 'labeled'` for them.
- **HDF5 export:** `synthetic_image_*` datasets are now labeled `units = counts/s`. Since
  v3.0.0 they were given the exposure's units instead (`Counts` or `DN/s` in v3.0.0; `DN`,
  `DN/s` or `MJy/sr` since), so **existing datasets with synthetic images carry the wrong
  label**. The pixel values were always counts/s; only the label was wrong. This fix doesn't
  change the exposure datasets' labels. Step 06 now refuses to export a synthetic
  image without a magnitude zeropoint, since its pixels wouldn't be in counts/s.
- `roman_test.yaml` used `psf.divide_up_detector: 5`, which `roman_util.divide_up_sca` rejects
  (the value must divide the 4088-pixel detector evenly), so step 00 failed. It now uses 2.
- `simple.yaml` and `roman_test.yaml` set `snr.snr_detector_position` to a position step 00
  caches PSFs at, so step 01b no longer needs STPSF data for the SNR PSF.
- **`snr.snr_detector_position` now defaults to the detector center, `(2043, 2043)`,**
  instead of the off-center `(2554, 2554)`. Only Roman PSFs depend on detector position, and
  every shipped Roman config sets the key, so the shipped configs are unaffected. But for your
  own Roman configs that leave it out, the SNR PSF in step 01b changes, so **which lenses pass
  the SNR cut can change**. To reproduce earlier results, set
  `snr_detector_position: [2554, 2554]`.
- `Pipeline.run()` no longer crashes in `calculate_snrs` with `imaging.engine: romanisim`. SNR
  exposures now always use GalSim, so `imaging.engine_params` must hold GalSim parameters even
  when `imaging.engine` is `romanisim`.
- **Step 00 now uses the same PSF cache as the later steps.** It used to join `psf_cache_dir`
  to `data_dir` itself, ignoring the fallback to the repository root and the default for
  `null`. So `simple.yaml` recomputed its PSFs with STPSF under `<data_dir>/tests/test_data`
  instead of using the ones in the repository, and `psf_cache_dir: null` crashed with a
  `TypeError`. New PSFs now go to the resolved directory, which can be `tests/test_data/` in
  your clone, or `mejiro/data/psfs/<instrument>` for `null`.
- The "Simplest Example" in the docs now runs: it passes `data_dir` and has the
  `if __name__ == '__main__':` guard that scripts need, because step 01b's workers import the
  script again.
- `Pipeline.print_metrics()` now reads from the `data_dir` given to `Pipeline`, finds `.npz`
  exposures as well as `.pkl`, and shows its results at any `logging_level`.
- `examples/run_mejiro_pipeline.py` now runs `simple.yaml` and takes the output directory as
  an argument, instead of writing to a hardcoded path.

### Dependencies

- Upgraded: lenstronomy 1.13.3 → 1.14.2, galsim 2.7.2 → 2.8.5, pyhalo 1.4.3 → 1.4.6, stpsf
  2.1.0 → 2.3.0, romanisim 0.13.0 → 0.14.0, romancal 0.22.0 → 1.1.0, colossus 1.3.10 →
  1.4.0, sncosmo 2.12.1 → 2.13.1. slsim, speclite and skypy are pinned to forks.
- pyhalo stays at 1.4.6: 1.4.7+ needs lenstronomy profiles that no release has yet.
- `numpy<2.4` is still required (slsim and pyhalo 1.4.6 use `np.trapz`).
- The `roman` extra now installs `roman-technical-information` from pip (no manual download
  or environment variable), and declares `pysiaf` and `astrocut`.

### Packaging and documentation

- **The version comes from git tags** (setuptools-scm). Between releases it names the exact
  commit, e.g. `3.0.1.dev12+g1a2b3c4`, so the `mejiro_version` attribute in HDF5 files
  identifies the code that produced them.
- The documentation was brought back in line with the code: the configuration reference,
  installation (which optional dependencies each use needs), the API reference, instrument
  support per pipeline step, and known issues. The module diagram is generated at build time.
- Tests in `tests/test_docs/` now fail when the docs drift from the code: undocumented config
  keys, missing API pages, or install instructions that disagree with `pyproject.toml`.
