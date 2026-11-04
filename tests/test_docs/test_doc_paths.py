"""Check that files the docs mention still exist.

When a script, config or example is renamed or deleted, prose that names it goes stale
silently. This test collects file paths mentioned in the Sphinx pages and the READMEs and
fails if any no longer exists in the repository.

A mention is a code literal that looks like a file path: ````...```` in reStructuredText,
```...``` or a link target in Markdown. A bare filename is found if it exists next to the
page, at the repository root, in ``docs/source/``, in ``mejiro/pipeline/`` or in
``mejiro/data/mejiro_config/``, since the docs often name scripts and configs without
their directory. Release notes and ``docs/dev/`` notes are dated records and aren't checked.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]

PAGES = sorted((REPO / 'docs' / 'source').rglob('*.rst')) + [
    REPO / 'README.md',
    REPO / 'docs' / 'README.md',
]

SEARCH_DIRS = [
    REPO,
    REPO / 'docs' / 'source',
    REPO / 'mejiro' / 'pipeline',
    REPO / 'mejiro' / 'data' / 'mejiro_config',
]

# Paths inside third-party packages, e.g. where a known-issues workaround says to edit.
THIRD_PARTY_PREFIXES = ('pyHalo/', 'lenstronomy/', 'slsim/', 'skypy/', 'speclite/')

EXTENSIONS = ('py', 'yaml', 'yml', 'ipynb', 'rst', 'md', 'toml', 'txt', 'sh', 'svg', 'png',
              'csv', 'cff', 'ecsv', 'npy', 'npz', 'pkl', 'h5')
LOOKS_LIKE_FILE = re.compile(r'[\w./-]+\.(?:%s)' % '|'.join(EXTENSIONS))
LOOKS_LIKE_DIR = re.compile(r'[\w.-]+(?:/[\w.-]+)*/')


def mentioned_paths(page):
    text = page.read_text()
    if page.suffix == '.rst':
        literals = re.findall(r'``([^`]+)``', text)
    else:
        literals = re.findall(r'`([^`\n]+)`', text) + re.findall(r'\]\(([^)\s]+)\)', text)
    for literal in literals:
        literal = literal.strip()
        if LOOKS_LIKE_FILE.fullmatch(literal) or LOOKS_LIKE_DIR.fullmatch(literal):
            yield literal


def exists(path, page):
    relative = path.lstrip('/')
    return any((base / relative).exists() for base in [page.parent, *SEARCH_DIRS])


def test_mentioned_files_exist():
    missing = sorted({
        f'{path} (in {page.relative_to(REPO)})'
        for page in PAGES
        for path in mentioned_paths(page)
        if not path.startswith(THIRD_PARTY_PREFIXES) and not exists(path, page)
    })
    assert not missing, (
        'Docs mention files that no longer exist: ' + ', '.join(missing)
        + '. Update the docs, or add a third-party prefix to THIRD_PARTY_PREFIXES.'
    )
