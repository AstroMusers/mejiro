"""Keep the installation page in sync with pyproject.toml and CI.

``docs/source/usage/installation.rst`` restates facts that live elsewhere. These tests
fail when:

1. the optional dependencies it describes don't match ``[project.optional-dependencies]``;
2. the Python versions it says are supported don't match the ``Programming Language ::
   Python :: 3.x`` classifiers, or the version pinned in its conda command isn't one of them;
3. the Python version it says CI tests isn't the one ``.github/workflows/python-app.yml``
   sets up.
"""
import pathlib
import re
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPO / 'docs' / 'source' / 'usage' / 'installation.rst'
PYPROJECT = tomllib.loads((REPO / 'pyproject.toml').read_text())
WORKFLOW = REPO / '.github' / 'workflows' / 'python-app.yml'


def page():
    return PAGE.read_text()


def classifier_versions():
    return {
        match.group(1)
        for classifier in PYPROJECT['project']['classifiers']
        if (match := re.fullmatch(r'Programming Language :: Python :: (3\.\d+)', classifier))
    }


def test_documented_extras_match_pyproject():
    # each optional dependency is described in a bullet that starts "- ``name``:"
    documented = set(re.findall(r'^- ``([\w-]+)``:', page(), re.MULTILINE))
    declared = set(PYPROJECT['project']['optional-dependencies'])
    assert documented == declared, (
        f'{PAGE.relative_to(REPO)} describes optional dependencies {sorted(documented)}, '
        f'but pyproject.toml declares {sorted(declared)}'
    )


def test_supported_python_versions_match_classifiers():
    match = re.search(r'supports Python ([\d. ,and]+?)\s*\(', page())
    assert match, 'expected a "supports Python ... (CI tests ...)" sentence'
    documented = set(re.findall(r'3\.\d+', match.group(1)))
    assert documented == classifier_versions(), (
        f'The page says Python {sorted(documented)} is supported, but the pyproject.toml '
        f'classifiers list {sorted(classifier_versions())}'
    )

    pinned = re.search(r'conda create -n mejiro python=(3\.\d+)', page())
    assert pinned, 'expected a "conda create -n mejiro python=3.x" command'
    assert pinned.group(1) in classifier_versions(), (
        f'The conda command pins Python {pinned.group(1)}, which the classifiers don\'t list'
    )


def test_ci_python_version_matches():
    documented = re.search(r'CI tests (3\.\d+)', page())
    assert documented, 'expected a "CI tests 3.x" statement'
    in_ci = re.findall(r'python-version:\s*["\']?(3\.\d+)', WORKFLOW.read_text())
    assert in_ci, f'no python-version found in {WORKFLOW.relative_to(REPO)}'
    assert documented.group(1) in in_ci, (
        f'The page says CI tests Python {documented.group(1)}, but the workflow uses {in_ci}'
    )
