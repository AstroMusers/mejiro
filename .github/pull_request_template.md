## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] Tests pass (`pytest`)
- [ ] Docs match the change. The tests in `tests/test_docs/` catch many of these, but not all:
  - [ ] Config keys added, removed or renamed: `docs/source/usage/configuration.rst` and the shipped configs in `mejiro/data/mejiro_config/`
  - [ ] Pipeline steps, flags or output files changed: `docs/source/usage/command_line.rst` and `docs/source/usage/outputs.rst`
  - [ ] Public modules added or removed: an `automodule` entry in `docs/source/mejiro/`, or a reason in `UNDOCUMENTED` in `tests/test_docs/test_api_coverage.py`
  - [ ] Dependencies or optional extras changed: `docs/source/usage/installation.rst`
- [ ] Changes users will notice are listed in `docs/releases/unreleased.md`
