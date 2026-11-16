## Unreleased (changes since v4.0.0)

Add to this file as changes land on `main`; rename it to `vX.Y.Z.md` at the next release.

### Pipeline

- **`exposure.usecrds`** (default `false`): set it to `true` and `_05_romanisim.py` takes
  romanisim's detector model from CRDS reference files (dark, flat, gain, read noise,
  linearity, saturation, IPC, distortion and the MA table) instead of romanisim's built-in
  defaults. The exposure time then comes from the CRDS MA table (and romanisim's read time is
  set to the table's frame time, without which romanisim 0.14 scales every L2 slope by their
  ratio), the lenses get the detector's relative QE, and `--level l3` places them with the
  CRDS-distortion WCS. Needs `CRDS_PATH` and
  `CRDS_SERVER_URL`. The reference files are downloaded before the batches start, and the
  context used is written to `05_romanisim/crds_context.txt` and to the HDF5 dataset's new
  `crds_context` attribute.
- `_05_romanisim.py --level l2` now writes `exposure_time.txt` as well, and `calculate_snrs.py`
  and `_06_h5_export.py` read the L2 exposure time from it. Earlier L2 runs, which don't have
  the file, still work.

### Bug fixes

- `make_substructure_csv.py` now takes the config with `--config` (and an optional `--data_dir`), like the other pipeline scripts. Before, it always read a hardcoded config from the repository's old `projects/` directory, which no longer exists.

### Dependencies

- The `roman` extra now requires `asdf-astropy>=0.11.0`. Older versions can't read the
  transforms in current CRDS reference files, so romanisim couldn't build the distortion WCS
  from them.

### Packaging and documentation

- New page: [Running on the Roman Research Nexus](https://mejiro.readthedocs.io/en/latest/usage/roman_research_nexus.html),
  covering installing `mejiro` in its own Nexus kernel, using the Nexus's CRDS cache and STPSF
  data, running the tests (including `pytest --run-crds`, which checks the CRDS setup), and
  injecting lenses into the SOC's HLWAS simulations.
