# Developer notes

Notes on upstream bugs, workarounds in mejiro, and proposed future work. Unlike the Sphinx docs,
these record what was true when they were written, so each one starts with two lines after its
title:

- `Status:` in one line: open (upstream or in mejiro), worked around (and where), fixed (and
  where), or proposal.
- `Last verified:` the date someone last checked the status, and against which versions.

When you check a note again, update both lines. When an issue is fixed and mejiro no longer
needs a workaround, either mark the note fixed or delete it (git history keeps it).

`tests/test_docs/test_dev_notes.py` checks that every note has both lines.
