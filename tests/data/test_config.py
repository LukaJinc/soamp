import pytest
from pydantic import ValidationError

from soamp.data.config import DatasetConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/data/base.yaml", DatasetConfig)
    assert cfg.output.filename == "mic_classification_dataset.csv"
    assert cfg.output.val_split_filename == "val_split.json"
    assert cfg.label_filter.included_labels == ["active", "inactive"]
    assert cfg.val_split.val_fraction == 0.3
    assert cfg.val_split.seed == 42
    assert cfg.val_split.identity_threshold == 0.60
    assert cfg.val_split.post_filtering is True
    assert cfg.paths.data_dir.name == "data"
    assert cfg.paths.data_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"paths": {"bogus_key": "x"}})


def test_val_split_extra_key_rejected():
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"val_split": {"bogus_key": "x"}})


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
