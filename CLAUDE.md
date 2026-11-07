# mejiro

Pipeline that simulates space-telescope images of galaxy-galaxy strong lenses (SLSim,
lenstronomy, pyHalo, GalSim, romanisim). Package code is in `mejiro/`, the numbered pipeline
steps in `mejiro/pipeline/`, shipped configs in `mejiro/data/mejiro_config/`, and Sphinx docs in
`docs/source/`.

## Keep the docs in sync

These rules and the tests in `tests/test_docs/` are there to stop the docs from drifting. 
When you change code, update the matching docs in the same change:

- **Config keys** added, removed or renamed: update the schema in
  `mejiro/pipeline/config_schema.py`, and the shipped configs in `mejiro/data/mejiro_config/`
  (remove keys nothing reads). The schema holds each key's type, default, description and the
  steps that need it; the reference in `configuration.rst` is generated from it at docs build
  time, so don't edit key entries there. Read keys directly (`config['key']`), since the schema
  owns the defaults. `test_config_reference.py` checks the schema against the code.
- **Pipeline steps, flags or output files**: update `docs/source/usage/command_line.rst` and
  `docs/source/usage/outputs.rst`. A new step also goes in `STEPS` in `config_schema.py`. The
  instrument list in the reference is generated from each step's `SUPPORTED_INSTRUMENTS`.
- **Modules** added, removed or renamed: add or remove the `automodule` entry in
  `docs/source/mejiro/`, or list the module in `UNDOCUMENTED` in `test_api_coverage.py` with the
  reason.
- **Import rules** changed in `pyproject.toml`: update the rules in
  `docs/source/appendices/structure.rst` to match. A new top-level module needs a layer there.
- **Dependencies, optional extras, or supported Python versions**: update
  `docs/source/usage/installation.rst` (`test_installation_docs.py`).
- **Code examples in the docs** live in `docs/source/examples/` and are included with
  `literalinclude`; `test_doc_examples.py` runs them, so don't paste runnable examples into
  `.rst` files. The exception is the end-to-end pipeline examples in
  `docs/source/usage/running.rst`, which are too slow to run as a test
  (`tests/test_pipeline/test_mejiro_pipeline.py` covers that path instead).
- **Changes users will notice**: add an entry to `docs/releases/unreleased.md`. Leave released
  notes (`docs/releases/vX.Y.Z.md`) alone.
- **Notes in `docs/dev/`** start with `Status:` and `Last verified:` lines (see
  `docs/dev/README.md`). Update them when the situation changes.

## Checks

- Docs guards (fast, no heavy dependencies): `pytest tests/test_docs --noconftest`
- Import rules (fast, no heavy dependencies beyond `import-linter`): `lint-imports`
- Full test suite (needs `pip install -e ".[dev,roman]"`): `pytest`. CI skips the HWO tests;
  see `.github/workflows/python-app.yml`.
- Docs build (needs graphviz): `pip install -r docs/requirements.txt`, then
  `sphinx-build -W --keep-going docs/source docs/build`

## Things to know

- The version comes from git tags via setuptools-scm (`mejiro/_version.py` is generated);
  don't hardcode it.
- `simple.yaml`, `test.yaml` and `roman_test.yaml` deliberately leave `data_dir` unset:
  `test_no_data_dir` relies on that, and the other tests pass a temporary `data_dir`.
- The HST instrument lives on the `feature/hst` branch, not on `main`.
- `config_schema.py` must import with only the standard library: Read the Docs imports it with
  none of mejiro's dependencies installed.
