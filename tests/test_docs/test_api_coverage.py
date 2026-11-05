"""Keep the API reference in sync with the package's modules.

The API pages under ``docs/source/`` list modules by hand in ``automodule`` directives,
and Sphinx says nothing when a module is missing. These tests fail when:

1. a module in ``mejiro/`` has no ``automodule`` entry and isn't in ``UNDOCUMENTED``;
2. an ``automodule`` entry names a module that no longer exists;
3. ``UNDOCUMENTED`` names a module that no longer exists, or one that is documented.

To fix a failure, add the module to the relevant page in ``docs/source/mejiro/``, or add it
to ``UNDOCUMENTED`` with the reason it's left out.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO / 'mejiro'
DOCS = REPO / 'docs' / 'source'

# Modules deliberately left out of the API reference, with the reason.
UNDOCUMENTED = {
    'mejiro.plots.corner': 'plotting helpers are not part of the documented API',
    'mejiro.plots.diagnostic_plot': 'plotting helpers are not part of the documented API',
    'mejiro.plots.overplot': 'plotting helpers are not part of the documented API',
    'mejiro.plots.plot': 'plotting helpers are not part of the documented API',
    'mejiro.style': 'internal matplotlib style helper',
    'mejiro.pipeline.make_substructure_csv': 'project-specific analysis script',
    'mejiro.engines.lenstronomy_engine': 'unreachable: Exposure raises NotImplementedError for this engine',
    'mejiro.engines.pandeia_engine': 'unreachable: Exposure raises NotImplementedError for this engine',
    'mejiro.utils.pipeline_util': 'unused duplicate of PipelineHelper.initialize_instrument_class',
    'mejiro.data.psfs.generate_psfs': 'maintenance script for the shipped PSFs',
}

# ".. automodule:: name" at the start of a line; commented-out entries (".. .. automodule::")
# don't match.
AUTOMODULE = re.compile(r'^\.\. automodule:: (\S+)', re.MULTILINE)


# Generated files that exist only after mejiro is installed (and are gitignored).
GENERATED = {PACKAGE / '_version.py'}  # written by setuptools-scm


def package_modules():
    """Dotted names of every module in mejiro/ (package ``__init__`` and generated files excluded)."""
    return {
        '.'.join(path.relative_to(REPO).with_suffix('').parts)
        for path in PACKAGE.rglob('*.py')
        if path.name != '__init__.py' and path not in GENERATED
    }


def documented_modules():
    """Map each module named in an ``automodule`` directive to the pages naming it."""
    found = {}
    for page in sorted(DOCS.rglob('*.rst')):
        for name in AUTOMODULE.findall(page.read_text()):
            found.setdefault(name, []).append(str(page.relative_to(REPO)))
    return found


def test_every_module_is_documented_or_deliberately_skipped():
    missing = sorted(package_modules() - set(documented_modules()) - set(UNDOCUMENTED))
    assert not missing, (
        'Modules with no automodule entry in docs/source/: ' + ', '.join(missing)
        + '. Document them, or add them to UNDOCUMENTED in this file with the reason.'
    )


def test_automodule_entries_name_existing_modules():
    modules = package_modules()
    stale = sorted(
        f'{name} (in {", ".join(pages)})'
        for name, pages in documented_modules().items()
        if name not in modules
    )
    assert not stale, 'automodule entries for modules that no longer exist: ' + ', '.join(stale)


def test_undocumented_list_is_current():
    modules, documented = package_modules(), set(documented_modules())
    gone = sorted(name for name in UNDOCUMENTED if name not in modules)
    assert not gone, 'UNDOCUMENTED lists modules that no longer exist: ' + ', '.join(gone)
    both = sorted(name for name in UNDOCUMENTED if name in documented)
    assert not both, 'UNDOCUMENTED lists modules that are documented: ' + ', '.join(both)
