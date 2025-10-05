import os
import json
import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def _safe_env(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")
    monkeypatch.setenv("CODECARBON_DISABLED", "1")
    monkeypatch.setenv("MLOPS_DISABLE_DAGSHUB", "1")
    yield

@pytest.fixture
def tmp_models_dir(tmp_path, monkeypatch):
    out = tmp_path / "models"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return out

@pytest.fixture
def tiny_classes_json(tmp_models_dir):
    data = {
        "class_to_id": {"Red":0,"Green":1,"Yellow":2,"off":3},
        "id_to_class": {"0":"Red","1":"Green","2":"Yellow","3":"off"},
    }
    p = tmp_models_dir / "classes.json"
    p.write_text(json.dumps(data, indent=2))
    return p

@pytest.fixture
def tiny_yaml(tmp_path):
    """Minimal dataset: 2 images, 2 boxes"""
    y = tmp_path / "train.yaml"
    (tmp_path / "rgb").mkdir(exist_ok=True)
    for name in ("a.png", "b.png"):
        (tmp_path / "rgb" / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    y.write_text("""
- path: rgb/a.png
  boxes:
    - x_min: 10
      y_min: 10
      x_max: 50
      y_max: 50
      label: Red
- path: rgb/b.png
  boxes:
    - x_min: 5
      y_min: 5
      x_max: 40
      y_max: 40
      label: Green
""")
    return y, tmp_path

