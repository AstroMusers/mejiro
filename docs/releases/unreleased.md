## Unreleased (changes since v4.0.0)

Add to this file as changes land on `main`; rename it to `vX.Y.Z.md` at the next release.

### Bug fixes

- `make_substructure_csv.py` now takes the config with `--config` (and an optional `--data_dir`), like the other pipeline scripts. Before, it always read a hardcoded config from the repository's old `projects/` directory, which no longer exists.
