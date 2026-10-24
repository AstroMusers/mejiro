import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture
def orchestrator(monkeypatch):
    script_names = (
        '_00_cache_psfs', '_01a_generate_galaxy_tables', '_01b_run_survey_simulation',
        '_02_build_lens_list', '_03_generate_subhalos', '_04_create_synthetic_images',
        '_05_galsim', '_06_h5_export', '_05_romanisim', 'calculate_snrs',
    )
    scripts = {}
    for name in script_names:
        qualified_name = 'mejiro.pipeline.' + name
        script = ModuleType(qualified_name)
        script.main = Mock()
        script.SCRIPT_NAME = name.removeprefix('_')
        script.DEFAULT_PATTERN = 'test-pattern'
        scripts[name] = script
        monkeypatch.setitem(sys.modules, qualified_name, script)
    path = Path(__file__).parents[2] / 'mejiro' / 'pipeline' / 'mejiro_pipeline.py'
    spec = importlib.util.spec_from_file_location('_mejiro_dispatch_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.util, 'print_execution_time', Mock())
    return module, scripts


@pytest.fixture
def make_pipeline(orchestrator, tmp_path):
    module, scripts = orchestrator

    def make(engine):
        config_file = tmp_path / 'dispatch.yaml'
        config_file.write_text(yaml.safe_dump({'imaging': {'engine': engine}}))
        return module.Pipeline(config_file=str(config_file), data_dir=str(tmp_path), _test_mode=True)

    return make


@pytest.mark.parametrize('engine', ['galsim', 'romanisim'])
@pytest.mark.parametrize('lens_step', ['02', '03'])
def test_complete_dispatch_keeps_lens_and_exposure_input_namespaces_separate(orchestrator, make_pipeline, engine, lens_step):
    module, scripts = orchestrator
    pipeline = make_pipeline(engine)
    pipeline.args.prev_step = lens_step
    order = []
    for name, script in scripts.items():
        script.main.side_effect = lambda args, name=name: order.append(name)
    pipeline.run()
    expected = ['_01a_generate_galaxy_tables', '_01b_run_survey_simulation',
                '_02_build_lens_list', '_03_generate_subhalos', '_04_create_synthetic_images',
                '_05_' + engine]
    if engine == 'romanisim':
        expected.append('calculate_snrs')
    assert order == expected + ['_06_h5_export']
    image_args = scripts['_04_create_synthetic_images'].main.call_args.args[0]
    exposure_args = scripts['_05_' + engine].main.call_args.args[0]
    export_args = scripts['_06_h5_export'].main.call_args.args[0]
    assert image_args is pipeline.args and exposure_args is pipeline.args
    assert image_args.prev_step == lens_step
    assert export_args is not pipeline.args and export_args.prev_step == '05_' + engine
    assert export_args.config == pipeline.args.config
    assert export_args.data_dir == pipeline.args.data_dir
    if engine == 'romanisim':
        snr_args = scripts['calculate_snrs'].main.call_args.args[0]
        assert snr_args.prev_step == '05_romanisim' and snr_args is export_args
    else:
        scripts['calculate_snrs'].main.assert_not_called()
    scripts['_00_cache_psfs'].main.assert_not_called()


@pytest.mark.parametrize('engine', ['galsim', 'romanisim'])
@pytest.mark.parametrize('step,script_name', [(6, '_06_h5_export'), ('snr', 'calculate_snrs')])
def test_individual_tail_steps_receive_exposure_inputs(orchestrator, make_pipeline, engine, step, script_name):
    module, scripts = orchestrator
    pipeline = make_pipeline(engine)
    original = vars(pipeline.args).copy()
    pipeline.run_script(step)
    dispatched = scripts[script_name].main.call_args.args[0]
    assert dispatched.prev_step == '05_' + engine
    assert vars(pipeline.args) == original


@pytest.mark.parametrize('engine', ['galsim', 'romanisim'])
def test_mutating_tail_arguments_does_not_modify_lens_input_step(orchestrator, make_pipeline, engine):
    module, scripts = orchestrator
    pipeline = make_pipeline(engine)
    original = vars(pipeline.args).copy()
    tail = pipeline._tail_args()
    assert isinstance(tail, argparse.Namespace)
    tail.prev_step = 'another_exposure_variant'
    assert vars(pipeline.args) == original


def test_invalid_engine_is_rejected_before_an_exposure_or_export_runs(orchestrator, make_pipeline):
    module, scripts = orchestrator
    pipeline = make_pipeline('unsupported')
    with pytest.raises(ValueError, match='Unsupported imaging engine'):
        pipeline._run_exposure_export_tail()
    for name in ('_05_galsim', '_05_romanisim', '_06_h5_export', 'calculate_snrs'):
        scripts[name].main.assert_not_called()