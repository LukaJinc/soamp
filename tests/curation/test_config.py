import pytest
from pydantic import ValidationError

from soamp.curation.config import CurationConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/curation/base.yaml", CurationConfig)
    assert cfg.split.identity_threshold == 0.60
    assert cfg.crawl.max_id == 24207
    assert cfg.paths.data_dir.name == "data"
    assert cfg.paths.data_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        CurationConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        CurationConfig.model_validate({"paths": {"bogus_key": "x"}})


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("split:\n  identity_threshold: 0.60\n")
    override = tmp_path / "experiment.yaml"
    override.write_text("_base_: base.yaml\nsplit:\n  identity_threshold: 0.80\n")
    cfg = load_config(override, CurationConfig)
    assert cfg.split.identity_threshold == 0.80
