"""Run the code examples shown in the documentation.

The examples in ``docs/source/examples/`` are pulled into the docs with ``literalinclude``,
so running them here means the docs can't show code that no longer works. An example with
a matching ``.txt`` file must print exactly that, since the docs show the ``.txt`` as the
example's output.

These need mejiro's full dependencies (``pip install -e ".[dev,roman]"``, as in CI) and are
skipped without them.
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DOCS = REPO / 'docs' / 'source'
EXAMPLES = DOCS / 'examples'
SCRIPTS = sorted(EXAMPLES.glob('*.py'))


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda path: path.name)
def test_example_runs(script, tmp_path):
    pytest.importorskip('galsim')
    pytest.importorskip('roman_technical_information')

    env = dict(os.environ, MPLBACKEND='Agg')  # examples may plot; don't open windows
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, f'{script.name} failed:\n{result.stderr[-3000:]}'

    expected_output = script.with_suffix('.txt')
    if expected_output.exists():
        assert result.stdout == expected_output.read_text(), (
            f'{script.name} printed something different from {expected_output.name}, which the '
            f'docs show as its output:\n{result.stdout}'
        )


def test_every_example_is_shown_in_the_docs():
    included = set()
    for page in DOCS.rglob('*.rst'):
        for target in re.findall(r'^\.\. literalinclude:: (\S+)', page.read_text(), re.MULTILINE):
            included.add((page.parent / target).resolve())
    unused = sorted(path.name for path in EXAMPLES.iterdir() if path.resolve() not in included)
    assert not unused, (
        f'Files in {EXAMPLES.relative_to(REPO)} that no page includes: ' + ', '.join(unused)
    )
