import pytest
from pydantic import ValidationError

from soamp.labeling.config import LabelingConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/labeling/base.yaml", LabelingConfig)
    assert cfg.threshold_table.filename == "organism_thresholds.csv"
    assert cfg.paths.data_dir.name == "data"
    assert cfg.paths.thresholds_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        LabelingConfig.model_validate({"paths": {"bogus_key": "x"}})


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "paths:\n  data_dir: data\n"
        "threshold_table:\n  filename: organism_thresholds.csv\n"
    )
    override = tmp_path / "experiment.yaml"
    override.write_text(
        "_base_: base.yaml\n"
        "threshold_table:\n  filename: alt_thresholds.csv\n"
    )
    cfg = load_config(override, LabelingConfig)
    assert cfg.threshold_table.filename == "alt_thresholds.csv"
    # unspecified-in-override field still comes from base.yaml
    assert cfg.paths.data_dir.name == "data"
