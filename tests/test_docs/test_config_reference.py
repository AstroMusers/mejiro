"""Keep the config schema in sync with the code that reads the config.

``mejiro/pipeline/config_schema.py`` is the one description of the config: it validates configs,
supplies defaults, and generates the reference in ``docs/source/usage/configuration.rst``. So
the docs can't drift, but the schema can. These tests fail when:

1. a shipped config (``mejiro/data/mejiro_config/*.yaml``) doesn't validate;
2. code reads a key that isn't in the schema;
3. a schema key isn't read anywhere in ``mejiro/pipeline/``;
4. a step reads a key the schema doesn't list for it, or the other way around;
5. code gives a config key its own default instead of leaving that to the schema;
6. the ``cores.script_<name>`` keys don't match the pipeline scripts. Those keys are read through
   an f-string (``f'script_{self.script_name}'``), so checks 2-4 can't see them;
7. the generated reference leaves a key out, or the schema can't be imported without mejiro's
   dependencies, which is how Read the Docs builds the docs.

If one of these fails, update ``config_schema.py`` (or the configs) to match the code.
"""
import ast
import collections
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

from mejiro.pipeline import config_schema
from mejiro.pipeline.config_schema import EVERY_STEP, KEYS, SCHEMA, STEPS, Need

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC = REPO / 'docs' / 'source' / 'usage' / 'configuration.rst'
EXT = REPO / 'docs' / 'source' / '_ext'
CONFIG_DIR = REPO / 'mejiro' / 'data' / 'mejiro_config'
PIPELINE = REPO / 'mejiro' / 'pipeline'

sys.path.insert(0, str(EXT))
import config_reference  # noqa: E402

# Variables that hold a section of the config. They reach worker functions by tuple unpacking,
# which the tracing below doesn't follow, so a new name for a section has to be added here.
SECTION_VARIABLES = {
    'survey_config': ['survey'],
    'snr_config': ['snr'],
    'psf_config': ['psf'],
    'imaging_config': ['imaging'],
    'synthetic_image_config': ['synthetic_image'],
    'subhalo_config': ['subhalos'],
    'dataset_config': ['dataset'],
}

Read = collections.namedtuple('Read', 'key kind line')


def _path(node, aliases):
    """The config path an expression points at, as a list of keys, or None if it isn't one."""
    if isinstance(node, ast.Name):
        return [] if node.id == 'config' else aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        return [] if node.attr == 'config' else None
    if isinstance(node, ast.Subscript):
        base = _path(node.value, aliases)
        if base is None:
            return None
        if isinstance(node.slice, ast.Constant):
            # a list index, as in psf.num_pixes[0], doesn't add a key
            return base + [node.slice.value] if isinstance(node.slice.value, str) else base
        return base + ['*']
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'get' and node.args):
        base = _path(node.func.value, aliases)
        if base is None:
            return None
        key = node.args[0]
        return base + [key.value if isinstance(key, ast.Constant) else '*']
    return None


def _scope_nodes(scope):
    """Nodes in a scope, in source order, not descending into nested functions."""
    nodes, stack = [], list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if not isinstance(node, (ast.FunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))
    return sorted(nodes, key=lambda node: getattr(node, 'lineno', 0))


def _trace(scope, aliases, found):
    aliases = dict(aliases)
    nodes = _scope_nodes(scope)
    for node in nodes:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name, path = node.targets[0].id, _path(node.value, aliases)
            if path is not None:
                aliases[name] = path
            elif name not in SECTION_VARIABLES:
                aliases.pop(name, None)
        elif isinstance(node, (ast.Subscript, ast.Call)):
            path = _path(node, aliases)
            if path:
                kind = 'get' if isinstance(node, ast.Call) else 'subscript'
                default = isinstance(node, ast.Call) and len(node.args) > 1
                found.append(Read('.'.join(path), 'default' if default else kind, node.lineno))
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            _trace(node, aliases, found)


def traced_reads(filename):
    """Every config read in a pipeline script."""
    found = []
    _trace(ast.parse((PIPELINE / filename).read_text()), SECTION_VARIABLES, found)
    return found


def read_keys(filename):
    """The dotted keys a pipeline script reads, leaving out whole sections and ``cores``."""
    return {
        read.key for read in traced_reads(filename)
        if read.key not in config_schema._CONTAINERS and not read.key.startswith('cores.')
    }


def pipeline_files():
    return sorted(path.name for path in PIPELINE.glob('*.py') if path.name != 'config_schema.py')


def steps_of(key):
    """The steps the schema lists for a key, without the ones covered by ``EVERY_STEP``."""
    return {need.step if isinstance(need, Need) else need for need in key.required_by + key.read_by}


def supported_instruments():
    return config_reference.supported_instruments()


@pytest.mark.parametrize('path', sorted(CONFIG_DIR.glob('*.yaml')), ids=lambda path: path.name)
def test_shipped_configs_validate(path):
    config = config_schema.apply_defaults(yaml.safe_load(path.read_text()))
    for step, script in STEPS.items():
        if config['instrument'] in supported_instruments().get(script, [config['instrument']]):
            for prev_step in ('05_galsim', '05_romanisim'):
                config_schema.validate(config, step, prev_step=prev_step, source=path.name)


def test_code_reads_only_schema_keys():
    unknown = sorted(
        f'{filename}:{read.line}: {read.key}'
        for filename in pipeline_files()
        for read in traced_reads(filename)
        if read.key not in KEYS and read.key not in config_schema._CONTAINERS
        and not read.key.startswith('cores.')
    )
    assert not unknown, 'Code reads config keys that aren\'t in config_schema.py: ' + ', '.join(unknown)


def test_every_schema_key_is_read():
    read = set().union(*(read_keys(filename) for filename in pipeline_files()))
    unread = sorted(name for name in KEYS if name not in read and not name.startswith('cores.'))
    assert not unread, (
        'Keys in config_schema.py that nothing in mejiro/pipeline/ reads: ' + ', '.join(unread)
        + '. Remove them from the schema and from the shipped configs.'
    )


def test_schema_lists_the_steps_that_read_each_key():
    helper_reads = read_keys('pipeline_helper.py')
    every_step = {
        name for name, key in KEYS.items() if EVERY_STEP in steps_of(key)
    }
    assert every_step == helper_reads, (
        'Keys marked EVERY_STEP should be exactly the ones PipelineHelper reads: '
        f'{sorted(every_step ^ helper_reads)}'
    )

    wrong = []
    for step, filename in STEPS.items():
        actual = read_keys(filename)
        listed = {
            name for name, key in KEYS.items()
            if step in steps_of(key) and not name.startswith('cores.')
        }
        wrong += [f'{filename} reads {name}, but the schema doesn\'t list step {step!r} for it'
                  for name in sorted(actual - listed - every_step)]
        # a key can also be read for a step through a PipelineHelper method
        wrong += [f'the schema lists step {step!r} for {name}, but {filename} doesn\'t read it'
                  for name in sorted(listed - actual - helper_reads)]
    assert not wrong, '; '.join(wrong)


def test_only_the_schema_sets_defaults():
    found = sorted(
        f'{filename}:{read.line}: {read.key}'
        for filename in pipeline_files()
        for read in traced_reads(filename)
        if read.kind in ('default', 'get')
    )
    assert not found, (
        'Read these keys as config[key]; defaults belong in config_schema.py: ' + ', '.join(found)
    )


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


def test_cores_keys_match_pipeline_scripts():
    scripts = pipeline_script_names()
    in_schema = {name.removeprefix('cores.script_') for name in KEYS if name.startswith('cores.')}

    unused = sorted(name for name in in_schema if not scripts.get(name, (None, False))[1])
    assert not unused, (
        'cores keys for scripts that don\'t exist or don\'t read a cores entry: '
        + ', '.join(f'cores.script_{name}' for name in unused)
    )

    missing = sorted(
        f'cores.script_{name} ({filename})'
        for name, (filename, reads_cores) in scripts.items()
        if reads_cores and name not in in_schema
    )
    assert not missing, 'Pipeline scripts that read a cores entry with no schema key: ' + ', '.join(missing)


def test_steps_match_the_pipeline_scripts():
    named = {name: filename for name, (filename, _) in pipeline_script_names().items()}
    numbered = {step: STEPS[step] for step in named if step in STEPS}
    assert numbered == named, 'STEPS in config_schema.py doesn\'t match the scripts\' SCRIPT_NAME'


def test_reference_documents_every_key():
    rendered = config_reference.render()
    for section in SCHEMA:
        assert section.title in rendered
        for key in section.keys:
            term = key.name.removeprefix(f'{section.name}.') if section.name else key.name
            assert re.search(rf'^``{re.escape(term)}``$', rendered, re.MULTILINE), key.name


def test_reference_lists_every_step_that_declares_instruments():
    rendered = config_reference.render()
    assert len(supported_instruments()) >= 10
    for script in supported_instruments():
        assert f'- ``{script}``: ' in rendered


def test_configuration_page_includes_the_generated_reference():
    assert '.. include:: ../_generated/config_reference.inc' in DOC.read_text()


def test_schema_and_reference_import_without_dependencies():
    """Read the Docs builds the docs with none of mejiro's dependencies installed."""
    code = (
        'import sys\n'
        f'sys.path[:0] = [{str(REPO)!r}, {str(EXT)!r}]\n'
        'import config_reference\n'
        'config_reference.render()\n'
        "heavy = {'numpy', 'yaml', 'astropy', 'galsim', 'lenstronomy', 'slsim', 'scipy'}\n"
        "print(sorted(heavy & {name.split('.')[0] for name in sys.modules}))\n"
    )
    # -S leaves out site-packages, so importing anything third-party fails
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'


def test_upgrade_notes_exist():
    """Error messages send people here, so renaming the file at a release has to update them."""
    notes = REPO / config_schema.UPGRADE_NOTES
    assert notes.exists(), f'UPGRADE_NOTES in config_schema.py names a missing file: {notes}'
    assert 'Upgrading from v3.0.0' in notes.read_text()
