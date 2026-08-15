import pytest
from pydantic import ValidationError

from soamp.features.config import FeaturesConfig
from soamp.features.peptide import DESCRIPTOR_NAMES
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/features/base.yaml", FeaturesConfig)
    assert cfg.descriptors.names == DESCRIPTOR_NAMES
    assert cfg.organism_vocab.unknown_index == 0
    assert cfg.paths.data_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"paths": {"bogus_key": "x"}})


def test_tracking_defaults():
    cfg = load_config("config/features/base.yaml", FeaturesConfig)
    assert cfg.tracking.exp_id == "features_pipeline"
    assert cfg.tracking.backend == "local"
    assert cfg.paths.tracking_dir.is_absolute()


def test_tracking_extra_key_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"tracking": {"bogus_key": "x"}})


def test_unknown_descriptor_name_rejected():
    with pytest.raises(ValidationError):
        FeaturesConfig.model_validate({"descriptors": {"names": ["NotARealDescriptor"]}})


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("descriptors:\n  names: [MolWt, TPSA]\n")
    override = tmp_path / "experiment.yaml"
    override.write_text("_base_: base.yaml\ndescriptors:\n  names: [MolWt]\n")
    cfg = load_config(override, FeaturesConfig)
    assert cfg.descriptors.names == ["MolWt"]
