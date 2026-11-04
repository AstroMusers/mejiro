# mejiro

Pipeline that simulates space-telescope images of galaxy-galaxy strong lenses (SLSim,
lenstronomy, pyHalo, GalSim, romanisim). Package code is in `mejiro/`, the numbered pipeline
steps in `mejiro/pipeline/`, shipped configs in `mejiro/data/mejiro_config/`, and Sphinx docs in
`docs/source/`.

## Keep the docs in sync

The docs drifted badly once; these rules and the tests in `tests/test_docs/` are there to stop
that happening again. When you change code, update the matching docs in the same change:

- **Config keys** added, removed or renamed: update `docs/source/usage/configuration.rst`, and
  the shipped configs in `mejiro/data/mejiro_config/` (remove keys nothing reads).
  `test_config_reference.py` checks both directions.
- **Pipeline steps, flags or output files**: update `docs/source/usage/command_line.rst` and
  `docs/source/usage/outputs.rst`. If a step's `SUPPORTED_INSTRUMENTS` changes, update the list
  under `instrument` in `configuration.rst` (also checked by `test_config_reference.py`).
- **Modules** added, removed or renamed: add or remove the `automodule` entry in
  `docs/source/mejiro/`, or list the module in `UNDOCUMENTED` in `test_api_coverage.py` with the
  reason.
- **Dependencies, optional extras, or supported Python versions**: update
  `docs/source/usage/installation.rst` (`test_installation_docs.py`).
- **Code examples in the docs** live in `docs/source/examples/` and are included with
  `literalinclude`; `test_doc_examples.py` runs them. Don't paste code into `.rst` files.
- **Changes users will notice**: add an entry to `docs/releases/unreleased.md`. Leave released
  notes (`docs/releases/vX.Y.Z.md`) alone.
- **Notes in `docs/dev/`** start with `Status:` and `Last verified:` lines (see
  `docs/dev/README.md`). Update them when the situation changes.

## Checks

- Docs guards (fast, no heavy dependencies): `pytest tests/test_docs --noconftest`
- Full test suite (needs `pip install -e ".[dev,roman]"`): `pytest`. CI skips the HWO tests;
  see `.github/workflows/python-app.yml`.
- Docs build (needs graphviz): `pip install -r docs/requirements.txt`, then
  `sphinx-build -W --keep-going docs/source docs/build`

## Things to know

- The version comes from git tags via setuptools-scm (`mejiro/_version.py` is generated);
  don't hardcode it.
- `simple.yaml` and `test.yaml` deliberately leave `data_dir` unset: `test_no_data_dir` relies
  on that, and the other tests pass a temporary `data_dir`.
- The HST and LSST instruments live on the `feature/hst` and `feature/lsst` branches, not on
  `main`.
