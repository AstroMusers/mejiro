from mejiro.pipeline.mejiro_pipeline import Pipeline


def test_pipeline_run(tmp_path):
    temp_dir = tmp_path / "subdir"
    temp_dir.mkdir()

    pipeline = Pipeline(data_dir=str(temp_dir), _test_mode=True)
    pipeline.run()

    # check script 01 output
    script_01_dir = temp_dir / "simple_dev" / "01"
    assert script_01_dir.exists() and script_01_dir.is_dir(), "Script 01 directory does not exist"

    detectable_lenses_pickle = list(script_01_dir.glob("detectable_gglenses_*.pkl"))
    assert len(detectable_lenses_pickle) == 1, f"Expected 1 .pkl file in {script_01_dir}, found {len(detectable_lenses_pickle)}"

    detectable_pop_csv = list(script_01_dir.glob("detectable_pop_*.csv"))
    assert len(detectable_pop_csv) == 1, f"Expected 1 .csv file in {script_01_dir}, found {len(detectable_pop_csv)}"

    total_pop_csv = list(script_01_dir.glob("total_pop_*.csv"))
    assert len(total_pop_csv) == 1, f"Expected 1 .csv file in {script_01_dir}, found {len(total_pop_csv)}"

    # check script 02 output
    script02_dir = temp_dir / "simple_dev" / "02"
    assert script02_dir.exists() and script02_dir.is_dir(), "Script 02 directory does not exist"

    script02_sca01_dir = script02_dir / "sca01"
    assert script02_sca01_dir.exists() and script02_sca01_dir.is_dir(), "sca01 directory does not exist in script 02 output"

    detectable_lens_pickles = list(script02_sca01_dir.glob("lens_simple_*.pkl"))
    assert len(detectable_lens_pickles) > 0, f"Expected at least 1 lens pickle file in {script02_sca01_dir}, found {len(detectable_lens_pickles)}"

    # check script 03 output
    script03_dir = temp_dir / "simple_dev" / "03"
    assert script03_dir.exists() and script03_dir.is_dir(), "Script 03 directory does not exist"
    
    script03_sca01_dir = script03_dir / "sca01"
    assert script03_sca01_dir.exists() and script03_sca01_dir.is_dir(), "sca01 directory does not exist in script 03 output"

    lens_with_subhalos_pickles = list(script03_sca01_dir.glob("lens_simple_*.pkl"))
    assert len(lens_with_subhalos_pickles) > 0, f"Expected at least 1 lens pickle file in {script03_sca01_dir}, found {len(lens_with_subhalos_pickles)}"

    subhalo_dir = script03_dir / "subhalos"
    assert subhalo_dir.exists() and subhalo_dir.is_dir(), "Subhalos directory does not exist in script 03 output"

    subhalo_pickles = list(subhalo_dir.glob("subhalo_realization_simple_*.pkl"))
    assert len(subhalo_pickles) > 0, f"Expected at least 1 subhalo pickle file in {subhalo_dir}, found {len(subhalo_pickles)}"

    # check script 04 output
    script04_dir = temp_dir / "simple_dev" / "04"
    assert script04_dir.exists() and script04_dir.is_dir(), "Script 04 directory does not exist"

    script04_sca01_dir = script04_dir / "sca01"
    assert script04_sca01_dir.exists() and script04_sca01_dir.is_dir(), "sca01 directory does not exist in script 04 output"

    synthetic_image_pickles = list(script04_sca01_dir.glob("SyntheticImage_simple_*.pkl"))
    assert len(synthetic_image_pickles) > 0, f"Expected at least 1 synthetic image pickle file in {script04_sca01_dir}, found {len(synthetic_image_pickles)}"

    # check script 05 output
    script05_dir = temp_dir / "simple_dev" / "05"
    assert script05_dir.exists() and script05_dir.is_dir(), "Script 05 directory does not exist"

    script05_sca01_dir = script05_dir / "sca01"
    assert script05_sca01_dir.exists() and script05_sca01_dir.is_dir(), "sca01 directory does not exist in script 05 output"

    exposure_pickles = list(script05_sca01_dir.glob("Exposure_simple_*.pkl"))
    assert len(exposure_pickles) > 0, f"Expected at least 1 exposure pickle file in {script05_sca01_dir}, found {len(exposure_pickles)}"

    # check script 06 output
    script06_dir = temp_dir / "simple_dev" / "06"
    assert script06_dir.exists() and script06_dir.is_dir(), "Script 06 directory does not exist"

    h5_file = list(script06_dir.glob("simple_v_0_1.h5"))
    assert len(h5_file) == 1, f"Expected 1 HDF5 file in {script06_dir}, found {len(h5_file)}"

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
