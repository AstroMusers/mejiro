## Unreleased (changes since v4.0.0)

Add to this file as changes land on `main`; rename it to `vX.Y.Z.md` at the next release.

### Bug fixes

- `make_substructure_csv.py` now takes the config with `--config` (and an optional `--data_dir`), like the other pipeline scripts. Before, it always read a hardcoded config from the repository's old `projects/` directory, which no longer exists.

### Dependencies

- The `roman` extra now requires `asdf-astropy>=0.11.0`. Older versions can't read the
  transforms in current CRDS reference files, so romanisim couldn't build the distortion WCS
  from them.
