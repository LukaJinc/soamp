import pytest
from pydantic import ValidationError

from soamp.data.config import DatasetConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/data/base.yaml", DatasetConfig)
    assert cfg.output.filename == "mic_classification_dataset.csv"
    assert cfg.label_filter.included_labels == ["active", "inactive"]
    assert cfg.paths.data_dir.name == "data"
    assert cfg.paths.data_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"paths": {"bogus_key": "x"}})


def test_tracking_defaults():
    cfg = load_config("config/data/base.yaml", DatasetConfig)
    assert cfg.tracking.exp_id == "dataset_pipeline"
    assert cfg.tracking.backend == "local"
    assert cfg.paths.tracking_dir.is_absolute()


def test_tracking_extra_key_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"tracking": {"bogus_key": "x"}})


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "paths:\n  data_dir: data\n"
        "label_filter:\n  included_labels: [active, inactive]\n"
    )
    override = tmp_path / "experiment.yaml"
    override.write_text(
        "_base_: base.yaml\n"
        "label_filter:\n  included_labels: [active]\n"
    )
    cfg = load_config(override, DatasetConfig)
    assert cfg.label_filter.included_labels == ["active"]
    assert cfg.paths.data_dir.name == "data"
