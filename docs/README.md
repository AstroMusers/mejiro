## Building documentation

To build docs locally:

1. Install the docs requirements: `pip install -r docs/requirements.txt`
2. Install graphviz, which draws the module diagram in `appendices/structure.rst` (e.g. `apt install graphviz`, `brew install graphviz` or `conda install graphviz`). The build stops with an error if graphviz's `dot` program isn't found.
3. From `docs/`, run `sphinx-build -M html source build`

## Release checklist

### Pre-Release

#### Dependencies

1. Grab the latest version of SLSim
2. Bump the pinned `roman-technical-information` version in `pyproject.toml` to the latest release
3. Update pinned git branches for forked dependencies in `pyproject.toml` (`speclite`, `skypy`)
4. Review and update all other dependency version constraints if needed

#### Code Quality

5. Ensure all tests pass locally: `pytest tests/`
6. Confirm CI passes on the `main` branch
7. Review and close or defer any open issues and pull requests intended for this release

#### Documentation

8. Update docstrings for any new or changed public API
9. Build docs locally and check for errors: `sphinx-build -W --keep-going docs/source docs/build`
10. Check `docs/releases/unreleased.md` covers the release, then rename it `docs/releases/vX.Y.Z.md` (and start a new, empty `docs/releases/unreleased.md`)
11. Update `CITATION.cff` with the new version number and release date

#### Versioning

12. Choose the version number following [semantic versioning](https://semver.org/). There's nothing to bump in the code: setuptools-scm takes the version from the git tag (see `[tool.setuptools_scm]` in `pyproject.toml`)

### Release

13. Commit the release notes and `CITATION.cff`
14. Tag the release: `git tag vX.Y.Z`
15. Push the tag: `git push origin vX.Y.Z`
16. Create a GitHub Release from the tag, using the release notes from step 10
17. Confirm the Zenodo integration mints a new DOI for the release
18. If `CITATION.cff`'s Zenodo identifier is a version-specific DOI, replace it with the new one (a concept DOI always resolves to the latest version and doesn't need changing)

### Post-Release

19. Confirm `pip install` from the tag reports the new version: `python -c "import mejiro; print(mejiro.__version__)"`
20. Confirm Read the Docs builds successfully for the new version (https://mejiro.readthedocs.io)
