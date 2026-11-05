"""Check that every note in docs/dev/ says how current it is.

Developer notes describe upstream bugs and workarounds at a point in time, so they go stale.
Each one must start with ``Status:`` and ``Last verified:`` lines (see docs/dev/README.md).
"""
import pathlib

import pytest

DEV_NOTES = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'dev'
NOTES = sorted(path for path in DEV_NOTES.glob('*.md') if path.name != 'README.md')


@pytest.mark.parametrize('note', NOTES, ids=lambda path: path.name)
def test_note_has_status_and_last_verified(note):
    head = note.read_text().splitlines()[:10]
    assert any(line.startswith('Status:') for line in head), f'{note.name} has no "Status:" line near the top'
    assert any(line.startswith('Last verified:') for line in head), f'{note.name} has no "Last verified:" line near the top'
