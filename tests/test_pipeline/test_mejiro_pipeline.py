import logging
import os

import h5py
import pytest
import yaml

import mejiro
from mejiro.pipeline.mejiro_pipeline import Pipeline
from mejiro.utils import util

TEST_CONFIG = os.path.join(os.path.dirname(mejiro.__file__), 'data', 'mejiro_config', 'test.yaml')


@pytest.mark.parametrize('serialization', ['full', 'lightweight'])
def test_pipeline_run(tmp_path, serialization):
    temp_dir = tmp_path / "subdir"
    temp_dir.mkdir()

    # derive a test config with the subhalos serialization mode under test
    with open(TEST_CONFIG) as f:
        config = yaml.safe_load(f)
    config['subhalos']['serialization'] = serialization
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

    pipeline = Pipeline(config_file=str(config_file), data_dir=str(temp_dir), _test_mode=True)
    pipeline.run()

    # check script 01a output
    script_01a_dir = temp_dir / "test_dev" / "01a"
    assert script_01a_dir.exists() and script_01a_dir.is_dir(), "Script 01a directory does not exist"

    galaxy_table_pickle = list(script_01a_dir.glob("galaxy_table_*.pkl"))
    assert len(galaxy_table_pickle) == 1, f"Expected 1 .pkl file in {script_01a_dir}, found {len(galaxy_table_pickle)}"

    # check script 01b output
    script_01b_dir = temp_dir / "test_dev" / "01b"
    assert script_01b_dir.exists() and script_01b_dir.is_dir(), "Script 01b directory does not exist"

    detectable_lenses_pickle = list(script_01b_dir.glob("detectable_gglenses_*.pkl"))
    assert len(detectable_lenses_pickle) == 1, f"Expected 1 .pkl file in {script_01b_dir}, found {len(detectable_lenses_pickle)}"

    sentinel_file = script_01b_dir / "run_complete_0000_sca01.txt"
    assert sentinel_file.exists(), f"Sentinel file {sentinel_file} does not exist, indicating script 01b did not complete successfully"
    with open(sentinel_file, 'r') as f:
        content = f.read()
        assert any(char.isdigit() for char in content), f"No digit found in sentinel file {sentinel_file}"

    detectable_pop_csv = list(script_01b_dir.glob("detectable_pop_*.csv"))
    assert len(detectable_pop_csv) == 1, f"Expected 1 .csv file in {script_01b_dir}, found {len(detectable_pop_csv)}"

    total_pop_csv = list(script_01b_dir.glob("total_pop_*.csv"))
    assert len(total_pop_csv) == 1, f"Expected 1 .csv file in {script_01b_dir}, found {len(total_pop_csv)}"

    # check script 02 output
    script02_dir = temp_dir / "test_dev" / "02"
    assert script02_dir.exists() and script02_dir.is_dir(), "Script 02 directory does not exist"

    script02_sca01_dir = script02_dir / "sca01"
    assert script02_sca01_dir.exists() and script02_sca01_dir.is_dir(), "sca01 directory does not exist in script 02 output"

    detectable_lens_pickles = list(script02_sca01_dir.glob("lens_test_*.pkl"))
    assert len(detectable_lens_pickles) > 0, f"Expected at least 1 lens pickle file in {script02_sca01_dir}, found {len(detectable_lens_pickles)}"

    # check script 03 output
    script03_dir = temp_dir / "test_dev" / "03"
    assert script03_dir.exists() and script03_dir.is_dir(), "Script 03 directory does not exist"
    
    script03_sca01_dir = script03_dir / "sca01"
    assert script03_sca01_dir.exists() and script03_sca01_dir.is_dir(), "sca01 directory does not exist in script 03 output"

    lens_with_subhalos_pickles = list(script03_sca01_dir.glob("lens_test_*.pkl"))
    assert len(lens_with_subhalos_pickles) > 0, f"Expected at least 1 lens pickle file in {script03_sca01_dir}, found {len(lens_with_subhalos_pickles)}"

    # verify the serialization mode on the step 03 pickle (fraction is 1.0, so this lens has substructure)
    lens = util.unpickle(str(lens_with_subhalos_pickles[0]))
    if serialization == 'lightweight':
        assert lens.realization == '<lightweight>', "lightweight 03 pickle should carry the sentinel"
        assert lens.substructure_flag == 'CDM', "lightweight 03 pickle should store the substructure flag"
    else:
        assert lens.realization is not None and not isinstance(lens.realization, str), \
            "full 03 pickle should retain the pyHalo realization object"

    # subhalo_dir = script03_dir / "subhalos"
    # assert subhalo_dir.exists() and subhalo_dir.is_dir(), "Subhalos directory does not exist in script 03 output"

    # subhalo_pickles = list(subhalo_dir.glob("subhalo_realization_simple_*.pkl"))
    # assert len(subhalo_pickles) > 0, f"Expected at least 1 subhalo pickle file in {subhalo_dir}, found {len(subhalo_pickles)}"

    # check script 04 output
    script04_dir = temp_dir / "test_dev" / "04"
    assert script04_dir.exists() and script04_dir.is_dir(), "Script 04 directory does not exist"

    script04_sca01_dir = script04_dir / "sca01"
    assert script04_sca01_dir.exists() and script04_sca01_dir.is_dir(), "sca01 directory does not exist in script 04 output"

    synthetic_image_pickles = list(script04_sca01_dir.glob("SyntheticImage_test_*.pkl"))
    assert len(synthetic_image_pickles) > 0, f"Expected at least 1 synthetic image pickle file in {script04_sca01_dir}, found {len(synthetic_image_pickles)}"

    # check script 05 output
    script05_dir = temp_dir / "test_dev" / "05_galsim"
    assert script05_dir.exists() and script05_dir.is_dir(), "Script 05 directory does not exist"

    script05_sca01_dir = script05_dir / "sca01"
    assert script05_sca01_dir.exists() and script05_sca01_dir.is_dir(), "sca01 directory does not exist in script 05 output"

    # step 05 writes '.npz' (imaging.serialization: lightweight) or '.pkl' ('full')
    exposure_ext = '.npz' if config['imaging']['serialization'] == 'lightweight' else '.pkl'
    exposure_files = list(script05_sca01_dir.glob(f"Exposure_test_*{exposure_ext}"))
    assert len(exposure_files) > 0, f"Expected at least 1 exposure {exposure_ext} file in {script05_sca01_dir}, found {len(exposure_files)}"

    # check script 06 output
    script06_dir = temp_dir / "test_dev" / "06"
    assert script06_dir.exists() and script06_dir.is_dir(), "Script 06 directory does not exist"

    h5_file = list(script06_dir.glob("test_v_0_1.h5"))
    assert len(h5_file) == 1, f"Expected 1 HDF5 file in {script06_dir}, found {len(h5_file)}"

    # the h5 is written even when no exposures are discovered, so assert on its contents:
    # test.yaml runs the galsim engine, so every exposure must be present and in DN
    with h5py.File(h5_file[0], 'r') as f:
        groups = list(f['images'])
        assert len(groups) > 0, "HDF5 'images' group is empty; step 06 found no exposures"

        band = config['synthetic_image']['bands'][0]
        for group_name in groups:
            uid = group_name.split('_')[-1]
            dset = f['images'][group_name][f'exposure_{uid}_{band}']
            assert dset.shape[0] > 0 and dset.shape[1] > 0, f"{group_name} exposure is empty"
            assert dset.attrs['units'][0] == 'DN', \
                f"galsim exposures should be in DN, got {dset.attrs['units'][0]}"
            # the galsim branch reads imaging.exposure_time; only romanisim input resolves
            # the exposure time from the step-05 sidecars
            assert float(dset.attrs['exposure_time'][0]) == float(config['imaging']['exposure_time']), \
                (f"galsim exposure_time should be imaging.exposure_time "
                 f"({config['imaging']['exposure_time']}), got {dset.attrs['exposure_time'][0]}")

            # test.yaml sets dataset.include_synthetic_images, and synthetic images are in
            # counts/s whatever the exposure's units (they used to inherit the exposure's)
            synth = f['images'][group_name][f'synthetic_image_{uid}_{band}']
            assert synth.attrs['units'][0] == 'counts/s', \
                f"synthetic images should be in counts/s, got {synth.attrs['units'][0]}"

def test_no_data_dir():
    try:
        pipeline = Pipeline()
        pipeline.run_script(0)
    except ValueError as e:
        assert str(e) == "data_dir must be specified either in the config file or via the --data_dir argument."
    else:
        assert False, "ValueError was not raised when data_dir was not specified"

def test_pipeline_invalid_script_number(tmp_path):
    temp_dir = tmp_path / "subdir"
    temp_dir.mkdir()

    pipeline = Pipeline(data_dir=str(temp_dir))
    try:
        pipeline.run_script(10)  # Invalid script number
    except ValueError as e:
        assert str(e) == "Script number 10 is not valid. Please choose a number between 0 and 6."
    else:
        assert False, "ValueError was not raised for invalid script number"


def test_tail_args_point_later_steps_at_the_exposure_step(tmp_path):
    # The orchestrator's --prev-step belongs to step 04 (which reads lens pickles from '02'
    # or '03'), but calculate_snrs and _06_h5_export read the same attribute to find
    # exposures. _tail_args() must hand those later steps a copy pointing at the exposure
    # step, and leave the args step 04 uses alone. See docs/dev/prev_step_namespace_collision.md.
    pipeline = Pipeline(config_file=TEST_CONFIG, data_dir=str(tmp_path), _test_mode=True)
    assert pipeline.args.prev_step == '03'

    pipeline._engine = 'galsim'
    assert pipeline._tail_args().prev_step == '05_galsim'

    pipeline._engine = 'romanisim'
    assert pipeline._tail_args().prev_step == '05_romanisim'

    assert pipeline.args.prev_step == '03', "_tail_args() must not modify the args step 04 uses"


@pytest.mark.parametrize('extension', ['.npz', '.pkl'])
def test_print_metrics_reads_the_data_dir_given_to_pipeline(tmp_path, caplog, extension):
    # simple.yaml leaves data_dir unset, so print_metrics has to use the one passed to Pipeline.
    # It also logs at WARNING, which print_metrics' results have to get through.
    pipeline_dir = tmp_path / 'simple_dev'
    exposure_dir = pipeline_dir / '05_galsim' / 'sca01'
    exposure_dir.mkdir(parents=True)
    (pipeline_dir / 'execution_times.json').write_text('{"01a": "0:00:10"}')
    for uid in ['00000000', '00000001']:
        (exposure_dir / f'Exposure_simple_{uid}_F106{extension}').touch()

    pipeline = Pipeline(data_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        pipeline.print_metrics()

    assert 'Execution time per system: 5.000 seconds' in caplog.text
