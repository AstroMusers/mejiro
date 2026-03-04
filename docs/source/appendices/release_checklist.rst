Release Checklist
=================

Pre-Release
-----------

Dependencies
^^^^^^^^^^^^

1. Grab the latest version of SLSim
2. Grab the latest version of ``roman-technical-information``
3. Update pinned git branches for forked dependencies in ``pyproject.toml`` (``speclite``, ``skypy``)
4. Review and update all other dependency version constraints if needed

Code Quality
^^^^^^^^^^^^

5. Ensure all tests pass locally: ``pytest tests/``
6. Confirm CI passes on the ``main`` branch
7. Review and close or defer any open issues and pull requests intended for this release

Documentation
^^^^^^^^^^^^^

8. Update docstrings for any new or changed public API
9. Generate the structure diagram with ``pydeps mejiro --only mejiro`` and update the image in ``docs/source/appendices/``
10. Build docs locally and check for errors: ``sphinx-build docs/source docs/build``
11. Update ``CITATION.cff`` with the new version number and release date

Versioning
^^^^^^^^^^

12. Bump the version number in ``mejiro/__init__.py`` following `semantic versioning <https://semver.org/>`_
13. Verify ``pyproject.toml`` picks up the new version dynamically

Release
-------

14. Commit the version bump: ``git commit -m "Bump version to vX.Y.Z"``
15. Tag the release: ``git tag vX.Y.Z``
16. Push the tag: ``git push origin vX.Y.Z``
17. Create a GitHub Release from the tag with a summary of changes
18. Confirm the Zenodo integration mints a new DOI for the release
19. Update the DOI badge in ``CITATION.cff`` and ``README`` with the new Zenodo DOI

Post-Release
------------

20. Confirm Read the Docs builds successfully for the new version (https://mejiro.readthedocs.io)
