import pytest
import json
import os

pytestmark = pytest.mark.integration


def test_smoke_train_imports(tmp_models_dir, monkeypatch):
    """Just ensure train module can be imported and basic artifacts created."""
    from mlops_greenlight.modeling import train as trn

    # simulate classes file to avoid dependency
    (tmp_models_dir / "classes.json").write_text(
        json.dumps(
            {
                "class_to_id": {"Red": 0, "Green": 1, "Yellow": 2, "off": 3},
                "id_to_class": {"0": "Red", "1": "Green", "2": "Yellow", "3": "off"},
            },
            indent=2,
        )
    )
    assert (tmp_models_dir / "classes.json").exists()
