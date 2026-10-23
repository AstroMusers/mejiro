import importlib
import os
from pathlib import Path

import mejiro.paths as paths


def test_repository_path_defaults_to_checkout(monkeypatch):
    monkeypatch.delenv("MEJIRO_PATH", raising=False)
    module = importlib.reload(paths)

    expected_path = Path(module.__file__).resolve().parents[1]
    assert os.environ["MEJIRO_PATH"] == str(expected_path)
    assert module.get_repository_path() == expected_path
    assert module.get_data_path() == expected_path / "data"
    assert module.get_visuals_path() == expected_path / "visuals"


def test_repository_path_honors_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MEJIRO_PATH", str(tmp_path))

    assert paths.get_repository_path() == tmp_path.resolve()