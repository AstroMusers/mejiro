"""Generate the key-by-key configuration reference from the schema in code.

``render()`` turns :data:`mejiro.pipeline.config_schema.SCHEMA` into reStructuredText. The
Sphinx hook writes that to ``_generated/config_reference.inc``, which
``usage/configuration.rst`` includes, so the reference can't disagree with the code. Like the
module diagram in ``conf.py``, it's generated on every build rather than committed.

This runs on Read the Docs, where none of mejiro's dependencies are installed, so it reads each
step's ``SUPPORTED_INSTRUMENTS`` from the script's source instead of importing the script.
"""
import ast
import inspect
import os
import pathlib
import textwrap

from mejiro.pipeline.config_schema import EVERY_STEP, SCHEMA, STEPS, Need

PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[3] / 'mejiro' / 'pipeline'

INSTRUMENT_NAMES = {'roman': 'Roman', 'jwst': 'JWST', 'hwo': 'HWO', 'lsst': 'LSST'}


def supported_instruments():
    """``SUPPORTED_INSTRUMENTS`` of each pipeline script that declares it, in step order."""
    found = {}
    for script in STEPS.values():
        for node in ast.parse((PIPELINE_DIR / script).read_text()).body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SUPPORTED_INSTRUMENTS' for t in node.targets)):
                found[script] = ast.literal_eval(node.value)
    return found


def _instrument_support():
    return '\n'.join(
        f'- ``{script}``: ' + ', '.join(INSTRUMENT_NAMES.get(name, name) for name in instruments)
        for script, instruments in supported_instruments().items()
    )


def _step(step):
    if step == EVERY_STEP:
        return 'every pipeline step'
    if step == 'pipeline':
        return '``Pipeline``'
    return f'``{STEPS[step]}``'


def _requirement(need):
    if isinstance(need, Need):
        return f'{_step(need.step)} when {need.when.text}'
    return _step(need)


def _facts(key):
    """The generated line under a key: its type, default, and which steps use it."""
    type_label = key.type.label
    if key.choices:
        type_label += ', one of ' + ', '.join(f'``{choice}``' for choice in key.choices)
    if key.nullable:
        type_label += ', or ``null``'
    facts = [f'*Type:* {type_label}.']
    if key.has_default:
        default = 'null' if key.default is None else key.default
        facts.append(f'*Default:* ``{default}``.')
    if key.runtime:
        facts.append('*Set at runtime.*')
    if key.required_by:
        facts.append('*Required by:* ' + '; '.join(_requirement(need) for need in key.required_by) + '.')
    if key.read_by:
        facts.append('*Read by:* ' + '; '.join(_step(step) for step in key.read_by) + '.')
    return ' '.join(facts)


def _render_key(key, section):
    term = key.name.removeprefix(f'{section.name}.') if section.name else key.name
    body = f'{inspect.cleandoc(key.description)}\n\n{_facts(key)}'
    return f'``{term}``\n{textwrap.indent(body, "   ")}'


def _render_section(section):
    parts = [f'{section.title}\n{"-" * len(section.title)}', inspect.cleandoc(section.intro)]
    parts += [_render_key(key, section) for key in section.keys]
    if section.outro:
        parts.append(inspect.cleandoc(section.outro).replace('{instrument_support}', _instrument_support()))
    return '\n\n'.join(parts)


def render():
    """The configuration reference, as reStructuredText."""
    return '\n\n'.join(_render_section(section) for section in SCHEMA) + '\n'


def write_reference(app):
    out_dir = os.path.join(app.srcdir, '_generated')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config_reference.inc'), 'w') as f:
        f.write(render())


def setup(app):
    app.connect('builder-inited', write_reference)
    return {'parallel_read_safe': True}
