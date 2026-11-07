"""Config validation and defaults (``mejiro.pipeline.config_schema``)."""
import argparse
import copy
import datetime
import logging
import os

import pytest
import yaml

import mejiro
from mejiro.pipeline import _00_cache_psfs
from mejiro.pipeline import config_schema
from mejiro.pipeline.config_schema import ConfigError, apply_defaults, validate
from mejiro.pipeline.mejiro_pipeline import Pipeline
from mejiro.pipeline.pipeline_helper import PipelineHelper, load_config

TEST_CONFIG = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'test.yaml')


@pytest.fixture
def config():
    with open(TEST_CONFIG) as f:
        return apply_defaults(yaml.safe_load(f))


def write(config, tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_missing_key_names_the_key_and_the_step(config):
    del config['subhalos']['serialization']
    with pytest.raises(ConfigError, match='missing `subhalos.serialization`, required by step 03'):
        validate(config, '03')


def test_a_step_only_needs_its_own_keys(config):
    del config['dataset']
    del config['subhalos']
    validate(config, '02')
    with pytest.raises(ConfigError, match='missing `dataset.labeled`, required by step 06'):
        validate(config, '06', prev_step='05_galsim')


def test_every_problem_is_reported_at_once(config):
    del config['subhalos']['fraction']
    del config['subhalos']['serialization']
    config['seed'] = 'forty-two'
    config['typo'] = 1
    with pytest.raises(ConfigError) as error:
        validate(config, '03')
    for text in ('subhalos.fraction', 'subhalos.serialization', '`seed`', 'unrecognized key `typo`'):
        assert text in str(error.value)


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_error_points_at_the_upgrade_notes(config):
    del config['subhalos']['serialization']
    with pytest.raises(ConfigError, match=f'"Upgrading from v3.0.0" in {config_schema.UPGRADE_NOTES}'):
        validate(config, '03')


@pytest.mark.parametrize('step, prev_step, message', [
    ('03', None, 'missing `subhalos.serialization`, required by step 03'),
    ('04', None, 'missing `synthetic_image.serialization`, required by step 04'),
    ('05_galsim', None, 'missing `imaging.serialization`, required by step 05_galsim'),
    ('05_galsim', None, 'missing `cores.script_05_galsim`, required by step 05_galsim'),
    ('06', '05_galsim', 'missing `dataset.labeled`, required by step 06'),
])
def test_v3_config_fails_with_the_documented_messages(config, step, prev_step, message):
    """Every change the upgrade notes ask for, undone."""
    v3 = copy.deepcopy(config)
    for section, key in (('subhalos', 'serialization'), ('synthetic_image', 'serialization'),
                         ('imaging', 'serialization'), ('dataset', 'labeled')):
        del v3[section][key]
    v3['cores']['script_05'] = v3['cores'].pop('script_05_galsim')
    with pytest.raises(ConfigError) as error:
        validate(v3, step, prev_step=prev_step)
    assert message in str(error.value)
    assert 'Upgrading from v3.0.0' in str(error.value)


@pytest.mark.parametrize('dotted', ['cores.script_01', 'cores.script_02', 'cores.script_05',
                                    'dataset.include_psfs', 'suppress_warnings'])
def test_keys_dropped_since_v3_are_unrecognized(config, dotted):
    *parents, name = dotted.split('.')
    mapping = config
    for parent in parents:
        mapping = mapping[parent]
    mapping[name] = 1
    with pytest.raises(ConfigError, match=f'unrecognized key `{dotted}`'):
        validate(config, '02')


def test_unknown_key_fails_for_any_step(config):
    config['synthetic_image']['oversampel'] = 4
    with pytest.raises(ConfigError, match='unrecognized key `synthetic_image.oversampel`'):
        validate(config, '00')


def test_nested_unknown_key(config):
    config['exposure']['coordinates']['roll'] = 0.
    with pytest.raises(ConfigError, match='unrecognized key `exposure.coordinates.roll`'):
        validate(config, '05_romanisim')


@pytest.mark.parametrize('section, key', [
    ('survey', 'catalog_source_kwargs'),
    ('survey', 'slhammocks_pipeline_kwargs'),
    ('subhalos', 'realization_kwargs'),
    ('imaging', 'engine_params'),
])
def test_pass_through_dicts_are_not_looked_inside(config, section, key):
    config[section][key]['anything_at_all'] = object()
    validate(config, '01b')


def test_remap_bands_contents_are_not_looked_inside(config):
    config['survey']['remap_bands'] = {'F106': 'F129', 'whatever': 3}
    validate(config, '02')


def test_a_section_must_be_a_mapping(config):
    config['psf'] = 3
    with pytest.raises(ConfigError, match='`psf`: expected a mapping, got 3'):
        validate(config, '02')


def test_empty_yaml_file():
    with pytest.raises(ConfigError, match='must be a YAML mapping'):
        apply_defaults(None)


@pytest.mark.parametrize('section, key, value, expected', [
    (None, 'dev', 'yes', 'boolean'),
    (None, 'seed', True, 'integer'),  # bool is an int in Python, but not here
    (None, 'seed', 1.5, 'integer'),
    (None, 'nice', None, 'integer'),
    ('survey', 'area', '0.5', 'number'),
    ('survey', 'bands', 'F106', 'list of strings'),
    ('survey', 'detectors', [1, 'two'], 'list of integers'),
    ('snr', 'snr_detector_position', [1, 2, 3], 'list of two numbers'),
    ('imaging', 'engine_params', 3, 'mapping'),
])
def test_wrong_type(config, section, key, value, expected):
    (config[section] if section else config)[key] = value
    with pytest.raises(ConfigError, match=f'expected {expected}'):
        validate(config, '02')


@pytest.mark.parametrize('section, key', [('subhalos', 'serialization'),
                                          ('synthetic_image', 'serialization'),
                                          ('imaging', 'serialization')])
def test_serialization_must_be_full_or_lightweight(config, section, key):
    config[section][key] = 'compressed'
    with pytest.raises(ConfigError, match=f"`{section}.{key}`: must be 'full' or 'lightweight'"):
        validate(config, '02')


def test_engine_must_be_a_known_simulator(config):
    config['imaging']['engine'] = 'pandeia'
    with pytest.raises(ConfigError, match="must be 'galsim' or 'romanisim'"):
        validate(config, '02')


@pytest.mark.parametrize('date', [datetime.datetime(2027, 4, 15), datetime.date(2027, 4, 15),
                                  '2027-04-15T00:00:00'])
def test_date_can_be_quoted_or_not(config, date):
    config['exposure']['date'] = date
    validate(config, '05_romanisim')


@pytest.mark.parametrize('version', [0.1, 1, '1.0.0'])
def test_version_can_be_a_number_or_a_string(config, version):
    config['dataset']['version'] = version
    validate(config, '06', prev_step='05_galsim')


def test_instrument_is_case_insensitive(config):
    config['instrument'] = 'Roman'
    del config['psf']['divide_up_detector']
    with pytest.raises(ConfigError, match='psf.divide_up_detector'):
        validate(config, '04')


def test_null_data_dir_is_allowed_in_the_file(config):
    assert config['data_dir'] is None
    validate(config, '02')


def test_data_dir_key_is_still_required(config):
    del config['data_dir']
    with pytest.raises(ConfigError, match='missing `data_dir`'):
        validate(config, '02')


def test_data_dir_needs_a_value_after_the_override(config):
    assert PipelineHelper.resolve_data_dir(config, '/some/dir') == '/some/dir'
    with pytest.raises(ValueError, match='^data_dir must be specified either in the config file '):
        PipelineHelper.resolve_data_dir(config)


def test_runtime_key_cant_be_set_in_a_file(config):
    config['survey']['cosmo'] = 'Planck18'
    with pytest.raises(ConfigError, match='`survey.cosmo` is set by the pipeline'):
        validate(config, '02')


# requirements that depend on an option

@pytest.mark.parametrize('switch, key, step', [
    ('survey.use_real_sources', 'survey.catalog_source_kwargs', '01b'),
    ('survey.use_slhammocks_pipeline', 'survey.slhammocks_pipeline_kwargs', '01a'),
    ('survey.use_slhammocks_pipeline', 'survey.deflector_z_min', '01a'),
    ('jaxtronomy.use_jax', 'jaxtronomy.jax_platform', '03'),
    ('jaxtronomy.use_jax', 'jaxtronomy.jax_platform', '04'),
    ('jaxtronomy.use_jax', 'jaxtronomy.jax_platform', 'snr'),
])
def test_required_only_with_a_switch(config, switch, key, step):
    section, name = key.split('.')
    del config[section][name]
    switch_section, switch_name = switch.split('.')
    config[switch_section][switch_name] = False
    validate(config, step, prev_step='05_galsim')
    config[switch_section][switch_name] = True
    with pytest.raises(ConfigError, match=f'missing `{key}`, required by step {step} when'):
        validate(config, step, prev_step='05_galsim')


def test_jax_platform_is_always_required_by_01b(config):
    del config['jaxtronomy']['jax_platform']
    with pytest.raises(ConfigError, match='missing `jaxtronomy.jax_platform`, required by step 01b'):
        validate(config, '01b')


def test_seed_is_required_by_03_only_for_partial_substructure(config):
    del config['seed']
    config['subhalos']['fraction'] = 1.
    validate(config, '03')
    config['subhalos']['fraction'] = 0.5
    with pytest.raises(ConfigError, match='missing `seed`, required by step 03 when'):
        validate(config, '03')


def test_exposure_section_only_for_romanisim(config):
    del config['exposure']
    for step in ('00', '01a', '01b', '02', '03', '04', '05_galsim'):
        validate(config, step)
    validate(config, 'snr', prev_step='05_galsim')
    validate(config, '06', prev_step='05_galsim')
    for step, prev_step in (('05_romanisim', None), ('snr', '05_romanisim'), ('06', '05_romanisim')):
        with pytest.raises(ConfigError, match='missing `exposure.ma_table_number`'):
            validate(config, step, prev_step=prev_step)


def test_exposure_time_only_for_galsim_input(config):
    del config['imaging']['exposure_time']
    validate(config, '06', prev_step='05_romanisim')
    with pytest.raises(ConfigError, match='missing `imaging.exposure_time`, required by step 06 when'):
        validate(config, '06', prev_step='05_galsim')


def test_divide_up_detector_only_for_l2(config):
    del config['psf']['divide_up_detector']
    validate(config, '05_romanisim', level='l3')
    with pytest.raises(ConfigError, match='required by step 05_romanisim when'):
        validate(config, '05_romanisim', level='l2')


def test_divide_up_detector_for_step_04_only_on_roman(config):
    del config['psf']['divide_up_detector']
    config['instrument'] = 'hwo'
    validate(config, '04')


def test_realization_kwargs_required_by_06_only_when_labeled(config):
    del config['subhalos']['realization_kwargs']
    config['dataset']['labeled'] = False
    validate(config, '06', prev_step='05_galsim')
    config['dataset']['labeled'] = True
    with pytest.raises(ConfigError, match='missing `subhalos.realization_kwargs`, required by step 06'):
        validate(config, '06', prev_step='05_galsim')


def test_project_script_only_needs_the_keys_every_step_needs(config):
    del config['dataset']
    validate(config, 'detectability')
    del config['pipeline_label']
    with pytest.raises(ConfigError, match='missing `pipeline_label`, required by step detectability'):
        validate(config, 'detectability')


def test_standalone_script_needs_only_what_it_reads():
    config = apply_defaults({'data_dir': None, 'pipeline_label': 'x', 'dev': False})
    validate(config, 'compact_03')


def test_remap_bands_warns_when_real_sources_are_on(config, caplog):
    config['survey']['use_real_sources'] = True
    with caplog.at_level(logging.WARNING):
        validate(config, '02')
    assert 'survey.remap_bands is not set' in caplog.text


def test_remap_bands_is_silent_when_set(config, caplog):
    config['survey']['use_real_sources'] = True
    config['survey']['remap_bands'] = {'F106': 'F106'}
    with caplog.at_level(logging.WARNING):
        validate(config, '02')
    assert 'remap_bands' not in caplog.text


# defaults: one test per former config.get(key, default)

@pytest.mark.parametrize('section, key, default', [
    (None, 'logging_level', 'INFO'),
    ('synthetic_image', 'oversample', 1),
    ('synthetic_image', 'deflector_only_fraction', 0.0),
    ('snr', 'snr_detector_position', (2043, 2043)),
    ('survey', 'remap_bands', None),
])
def test_default_matches_the_old_get(config, section, key, default):
    mapping = config[section] if section else config
    mapping.pop(key, None)
    apply_defaults(config)
    assert mapping[key] == default
    assert type(mapping[key]) is type(default)


def test_defaults_are_the_only_ones():
    defaults = {name for name, key in config_schema.KEYS.items() if key.has_default}
    assert defaults == {'logging_level', 'synthetic_image.oversample',
                        'synthetic_image.deflector_only_fraction', 'snr.snr_detector_position',
                        'survey.remap_bands'}


@pytest.mark.parametrize('section, key', [(None, 'nice'), ('survey', 'num_galaxy_tables'),
                                          ('survey', 'speed_factor'), ('jaxtronomy', 'jax_platform')])
def test_keys_that_lost_their_default_since_v3_stay_required(config, section, key):
    mapping = config[section] if section else config
    del mapping[key]
    apply_defaults(config)
    assert key not in mapping


def test_set_values_are_kept(config):
    config['synthetic_image']['oversample'] = 4
    config['logging_level'] = 'ERROR'
    apply_defaults(config)
    assert config['synthetic_image']['oversample'] == 4
    assert config['logging_level'] == 'ERROR'


def test_explicit_null_stays_null_like_dict_get(config):
    config['survey']['remap_bands'] = None
    config['synthetic_image']['oversample'] = None
    apply_defaults(config)
    assert config['synthetic_image']['oversample'] is None


def test_defaults_dont_create_sections():
    config = apply_defaults({'pipeline_label': 'x'})
    assert 'synthetic_image' not in config and 'snr' not in config


def test_load_config_fills_defaults(tmp_path, config):
    del config['synthetic_image']['oversample']
    assert load_config(write(config, tmp_path))['synthetic_image']['oversample'] == 1


def test_load_config_finds_the_extension(tmp_path, config):
    path = write(config, tmp_path)
    assert load_config(path.removesuffix('.yaml'))['pipeline_label'] == 'test'


# validation in the pipeline itself

def test_step_stops_before_touching_the_disk(tmp_path, config):
    del config['psf']['bands']
    args = argparse.Namespace(config=write(config, tmp_path), data_dir=str(tmp_path / 'out'))
    with pytest.raises(ConfigError, match='missing `psf.bands`, required by step 00'):
        _00_cache_psfs.main(args)
    assert not (tmp_path / 'out').exists()


def test_pipeline_requires_the_engine(tmp_path, config):
    del config['imaging']['engine']
    with pytest.raises(ConfigError, match='missing `imaging.engine`, required by `Pipeline`'):
        Pipeline(config_file=write(config, tmp_path), data_dir=str(tmp_path))


def test_run_checks_every_step_before_the_first(tmp_path, config):
    del config['dataset']['labeled']
    pipeline = Pipeline(config_file=write(config, tmp_path), data_dir=str(tmp_path / 'out'), _test_mode=True)
    with pytest.raises(ConfigError, match='missing `dataset.labeled`, required by step 06'):
        pipeline.run()
    assert not (tmp_path / 'out').exists()


def test_run_only_checks_the_chosen_engine(tmp_path, config):
    """A galsim run doesn't need romanisim's keys."""
    del config['exposure']
    pipeline = Pipeline(config_file=write(config, tmp_path), data_dir=str(tmp_path), _test_mode=True)
    pipeline._validate_run()

    config['imaging']['engine'] = 'romanisim'
    pipeline = Pipeline(config_file=write(config, tmp_path), data_dir=str(tmp_path), _test_mode=True)
    with pytest.raises(ConfigError, match='missing `exposure.date`, required by step 05_romanisim'):
        pipeline._validate_run()


def test_run_without_data_dir(tmp_path, config):
    pipeline = Pipeline(config_file=write(config, tmp_path), _test_mode=True)
    with pytest.raises(ValueError, match='^data_dir must be specified either in the config file '):
        pipeline.run()
