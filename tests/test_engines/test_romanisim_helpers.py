import ast
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from astropy.table import Table

from tdpy.verbosity import print


MEJIRO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = MEJIRO_ROOT / "mejiro" / "engines" / "romanisim_engine.py"
ROMANISIM_NOTEBOOKS = (
    MEJIRO_ROOT / "dev" / "engines" / "romanisim" / "romanisim_explain.ipynb",
    MEJIRO_ROOT / "dev" / "engines" / "romanisim" / "romanisim_ffi.ipynb",
    MEJIRO_ROOT / "dev" / "engines" / "romanisim" / "romanisim_inject_lens.ipynb",
)


def test_romanisim_simulators_preserve_inputs_and_restore_crds_reference_data(monkeypatch):
    print(f"Reading from {ENGINE_PATH}...")
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"), filename=str(ENGINE_PATH))
    function_names = {"_simulate_lens", "simulate_lens_no_crds", "simulate_lens_with_crds"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {function.name for function in functions} == function_names
    defaults = {
        "instrument": {"detector": "WFI01", "optical_element": "F158"},
        "exposure": {"ma_table_number": 4, "read_pattern": "default", "start_time": None},
    }
    calls = []
    coordinate = object()
    extra_counts = np.ones((2, 2))
    time_value = object()
    random_generator = object()
    result = (object(), {"metadata": "preserved"})
    read_pattern = [[0, 1], [1, 2]]
    reference_data = {"gain": 2.0, "zeropoint": 25.0}
    galsim_module = ModuleType("galsim")
    galsim_module.UniformDeviate = lambda seed: (seed, random_generator)
    monkeypatch.setitem(sys.modules, "galsim", galsim_module)

    def simulate(meta, catalog, **kwargs):
        calls.append((meta, catalog, kwargs, dict(reference_data)))
        return result

    namespace = {
        "deepcopy": deepcopy,
        "Time": lambda value, format: (value, format) if value != "observed" else time_value,
        "Table": Table,
        "np": np,
        "rsim_params": SimpleNamespace(
            default_parameters_dictionary=defaults,
            read_pattern={4: read_pattern},
            reference_data=reference_data,
        ),
        "rsim_wcs": SimpleNamespace(
            fill_in_parameters=lambda meta, coord, boresight: meta.update(
                {"boresight": (coord, boresight)}
            )
        ),
        "rsim_image": SimpleNamespace(simulate=simulate),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(ENGINE_PATH), "exec"),
        namespace,
    )

    actual = namespace["simulate_lens_no_crds"](
        extra_counts, "F158", 3, 4, coordinate, "observed", rng_seed=9
    )

    assert actual is result
    assert defaults["instrument"]["detector"] == "WFI01"
    assert reference_data == {"gain": 2.0, "zeropoint": 25.0}
    assert len(calls) == 1
    meta, catalog, options, references_during_simulation = calls[0]
    assert meta["instrument"]["detector"] == "WFI03"
    assert meta["instrument"]["optical_element"] == "F158"
    assert meta["exposure"]["read_pattern"] is read_pattern
    assert meta["exposure"]["start_time"] is time_value
    assert meta["boresight"] == (coordinate, True)
    assert isinstance(catalog, Table)
    assert catalog.colnames == ["ra", "dec", "type", "n", "half_light_radius", "pa", "ba", "F158"]
    assert len(catalog) == 0
    assert references_during_simulation == reference_data
    assert options == {
        "usecrds": False,
        "psftype": "galsim",
        "level": 2,
        "rng": (9, random_generator),
        "crparam": {},
        "extra_counts": extra_counts,
    }

    supplied_catalog = Table({"ra": [1.0]})
    namespace["simulate_lens_no_crds"](
        extra_counts, "F158", 3, 4, coordinate, "observed",
        level=1, crparam=None, catalog=supplied_catalog,
    )
    _, actual_catalog, options, references_during_simulation = calls[1]
    assert actual_catalog is supplied_catalog
    assert options["level"] == 1
    assert options["crparam"] is None
    assert references_during_simulation == reference_data

    namespace["simulate_lens_with_crds"](
        extra_counts, "F158", 3, 4, coordinate, "observed", rng_seed=9
    )
    _, _, crds_options, references_during_simulation = calls[2]
    assert crds_options["usecrds"] is True
    assert crds_options["psftype"] == "epsf"
    assert references_during_simulation == {"gain": None, "zeropoint": None}
    assert reference_data == {"gain": 2.0, "zeropoint": 25.0}

    def fail_simulation(*args, **kwargs):
        raise RuntimeError("simulated CRDS failure")

    namespace["rsim_image"].simulate = fail_simulation
    try:
        namespace["simulate_lens_with_crds"](
            extra_counts, "F158", 3, 4, coordinate, "observed"
        )
    except RuntimeError as error:
        assert str(error) == "simulated CRDS failure"
    else:
        raise AssertionError("CRDS simulation failure was not propagated")
    assert reference_data == {"gain": 2.0, "zeropoint": 25.0}


def test_poisson_stamp_helper_clips_to_detector_and_reuses_seed(monkeypatch):
    import ast

    print(f"Reading from {ENGINE_PATH}...")
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"), filename=str(ENGINE_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "poisson_stamp_to_extra_counts"
    )
    image_type = type("Image", (), {"__init__": lambda self, array: setattr(self, "array", array)})
    galsim_module = ModuleType("galsim")
    galsim_module.roman = SimpleNamespace(n_pix=3)
    galsim_module.ImageI = image_type
    monkeypatch.setitem(sys.modules, "galsim", galsim_module)
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(ENGINE_PATH), "exec"),
        namespace,
    )
    stamp = np.array([[2.5, -1.0], [1.0, 3.0]])
    original_stamp = stamp.copy()

    centered = namespace["poisson_stamp_to_extra_counts"](
        stamp, center_x=1, center_y=1, rng_seed=42
    )
    repeated = namespace["poisson_stamp_to_extra_counts"](
        stamp, center_x=1, center_y=1, rng_seed=42
    )
    edge = namespace["poisson_stamp_to_extra_counts"](
        stamp, center_x=0, center_y=0, rng_seed=42
    )

    expected_rng = np.random.default_rng(42)
    expected_center = np.zeros((3, 3), dtype=np.int32)
    expected_center[:2, :2] = expected_rng.poisson(np.clip(stamp, 0, None)).astype(np.int32)
    expected_edge = np.zeros((3, 3), dtype=np.int32)
    expected_edge[0, 0] = np.random.default_rng(42).poisson(3.0)
    np.testing.assert_array_equal(centered.array, expected_center)
    np.testing.assert_array_equal(repeated.array, expected_center)
    np.testing.assert_array_equal(edge.array, expected_edge)
    np.testing.assert_array_equal(stamp, original_stamp)


def test_romanisim_notebooks_import_the_shared_no_crds_simulator():
    expected_no_crds_source = "from mejiro.engines.romanisim_engine import simulate_lens_no_crds"
    expected_crds_source = "from mejiro.engines.romanisim_engine import simulate_lens_with_crds"
    expected_count_source = (
        "from mejiro.engines.romanisim_engine import "
        "poisson_stamp_to_extra_counts as build_extra_counts"
    )
    for path in ROMANISIM_NOTEBOOKS:
        print(f"Reading from {path}...")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_sources = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        assert expected_no_crds_source in code_sources
        assert expected_crds_source in code_sources
        assert not any("def simulate_lens_no_crds" in source for source in code_sources)
        assert not any("def simulate_lens_with_crds" in source for source in code_sources)
        matching_cells = [
            cell for cell in notebook["cells"]
            if cell.get("cell_type") == "code" and cell.get("id") == "c0397a4e"
        ]
        assert len(matching_cells) == 1
        assert "".join(matching_cells[0]["source"]).strip() == expected_no_crds_source
        if path.name in {"romanisim_explain.ipynb", "romanisim_inject_lens.ipynb"}:
            assert expected_count_source in code_sources
            assert any("build_extra_counts(" in source for source in code_sources)
            assert not any("def build_extra_counts" in source for source in code_sources)