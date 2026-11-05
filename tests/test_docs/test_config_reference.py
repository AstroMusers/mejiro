"""Keep the configuration reference in sync with the shipped configs and the code.

``docs/source/usage/configuration.rst`` is written by hand, so nothing stops it from
drifting when config keys are added, renamed or dropped. These tests fail when:

1. a key in a shipped config (``mejiro/data/mejiro_config/*.yaml``) isn't documented;
2. a documented key isn't read anywhere in ``mejiro/`` (its name never appears as a
   string in the code), so the page describes a key that no longer does anything;
3. the ``cores.script_<name>`` keys don't match the pipeline scripts. Those keys are
   read through an f-string (``f'script_{self.script_name}'``), so check 2 can't see them;
4. the list of which instruments each pipeline step supports (in the ``instrument``
   section) doesn't match the steps' ``SUPPORTED_INSTRUMENTS``.

If one of these fails, update ``configuration.rst`` (or the configs) to match the code.
"""
import ast
import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC = REPO / 'docs' / 'source' / 'usage' / 'configuration.rst'
CONFIG_DIR = REPO / 'mejiro' / 'data' / 'mejiro_config'
PACKAGE = REPO / 'mejiro'
PIPELINE = PACKAGE / 'pipeline'

# Dicts whose contents are handed straight to another package (slsim, pyHalo, the
# detector-effects engine). The reference documents the dict itself, not its keys.
PASSTHROUGH = {
    'survey.catalog_source_kwargs',
    'survey.slhammocks_pipeline_kwargs',
    'survey.remap_bands',
    'subhalos.realization_kwargs',
    'imaging.engine_params',
}

CORES_KEY = re.compile(r'cores\.script_(\w+)')


def documented_keys():
    """Dotted names of every key documented in configuration.rst.

    A section title is a ````name```` line underlined with dashes; the definition-list
    terms beneath it (````key```` at column 0, possibly several per line) are its keys.
    Terms under the "Global attributes" section are top-level keys.
    """
    lines = DOC.read_text().splitlines()
    keys, section = set(), None
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ''
        if re.fullmatch(r'-{3,}', next_line):
            match = re.fullmatch(r'``([\w.]+)``', line.strip())
            section = match.group(1) if match else None
            if section:
                keys.add(section)
            continue
        if line.startswith('``'):
            for name in re.findall(r'``([\w.]+)``', line):
                keys.add(f'{section}.{name}' if section else name)
    return keys


def _flatten(mapping, prefix=''):
    for key, value in mapping.items():
        dotted = f'{prefix}{key}'
        yield dotted
        if isinstance(value, dict) and dotted not in PASSTHROUGH:
            yield from _flatten(value, f'{dotted}.')


def shipped_config_keys():
    """Map each dotted key in the shipped configs to the configs that set it."""
    found = {}
    for path in sorted(CONFIG_DIR.glob('*.yaml')):
        for key in _flatten(yaml.safe_load(path.read_text())):
            found.setdefault(key, []).append(path.name)
    return found


def string_constants():
    """Every string constant in the package's code (keys are read as ``config['key']``)."""
    constants = set()
    for path in PACKAGE.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                constants.add(node.value)
    return constants


def pipeline_script_names():
    """``SCRIPT_NAME`` of each pipeline script, and whether it reads a ``cores`` entry."""
    scripts = {}
    for path in sorted(PIPELINE.glob('*.py')):
        source = path.read_text()
        for node in ast.parse(source).body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SCRIPT_NAME' for t in node.targets)
                    and isinstance(node.value, ast.Constant)):
                reads_cores = 'calculate_process_count' in source or "['cores']" in source
                scripts[node.value.value] = (path.name, reads_cores)
    return scripts


def test_shipped_config_keys_are_documented():
    documented = documented_keys()
    undocumented = {
        key: configs for key, configs in shipped_config_keys().items()
        # a dict like exposure.coordinates counts as documented through its children
        if key not in documented and not any(d.startswith(f'{key}.') for d in documented)
    }
    assert not undocumented, (
        f'Keys in shipped configs but missing from {DOC.relative_to(REPO)}: '
        + ', '.join(f'{key} (in {", ".join(configs)})' for key, configs in sorted(undocumented.items()))
    )


def test_documented_keys_are_read_by_code():
    constants = string_constants()
    unread = sorted(
        key for key in documented_keys()
        if not CORES_KEY.fullmatch(key) and key.rsplit('.', 1)[-1] not in constants
    )
    assert not unread, (
        f'Keys documented in {DOC.relative_to(REPO)} that nothing in mejiro/ reads: '
        + ', '.join(unread)
        + '. Remove them from the docs (and from the shipped configs) or fix the reference.'
    )


def test_cores_keys_match_pipeline_scripts():
    scripts = pipeline_script_names()
    documented = {m.group(1) for m in map(CORES_KEY.fullmatch, documented_keys()) if m}

    unused = sorted(name for name in documented if not scripts.get(name, (None, False))[1])
    assert not unused, (
        'cores keys documented for scripts that don\'t exist or don\'t read a cores entry: '
        + ', '.join(f'cores.script_{name}' for name in unused)
    )

    missing = sorted(
        f'cores.script_{name} ({filename})'
        for name, (filename, reads_cores) in scripts.items()
        if reads_cores and name not in documented
    )
    assert not missing, (
        f'Pipeline scripts that read a cores entry with no documented key in '
        f'{DOC.relative_to(REPO)}: ' + ', '.join(missing)
    )


# "- ``_00_cache_psfs.py``: Roman" -- one line per pipeline script, at column 0
SUPPORT_LINE = re.compile(r'^- ``(\w+\.py)``: (.+)$', re.MULTILINE)


def documented_instrument_support():
    """Map each pipeline script in the ``instrument`` section's list to its instruments."""
    return {
        script: {name.strip().lower() for name in instruments.split(',')}
        for script, instruments in SUPPORT_LINE.findall(DOC.read_text())
    }


def supported_instruments():
    """``SUPPORTED_INSTRUMENTS`` of each pipeline script that declares it."""
    found = {}
    for path in sorted(PIPELINE.glob('*.py')):
        for node in ast.parse(path.read_text()).body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SUPPORTED_INSTRUMENTS' for t in node.targets)):
                found[path.name] = set(ast.literal_eval(node.value))
    return found


def test_instrument_support_list_matches_pipeline_scripts():
    documented, actual = documented_instrument_support(), supported_instruments()
    wrong = sorted(
        f'{script}: docs say {sorted(documented.get(script, set()))}, code says {sorted(actual.get(script, set()))}'
        for script in documented.keys() | actual.keys()
        if documented.get(script) != actual.get(script)
    )
    assert not wrong, (
        f'The instrument support list in {DOC.relative_to(REPO)} is out of date: ' + '; '.join(wrong)
    )
