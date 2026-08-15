import pytest
from pydantic import ValidationError

from soamp.engine.config import TrainConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/train/base.yaml", TrainConfig)
    assert cfg.tracking.exp_id == "baseline_mlp_v1"
    assert cfg.class_balancing.mode == "auto"
    assert cfg.paths.checkpoint_dir.is_absolute()


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"bogus_top_level_key": 1})


def test_extra_nested_key_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"loop": {"bogus_key": 1}})


def test_class_balancing_fixed_mode_without_value_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"class_balancing": {"mode": "fixed"}})


def test_class_balancing_unknown_mode_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"class_balancing": {"mode": "bogus"}})


def test_class_balancing_fixed_mode_with_value_accepted():
    cfg = TrainConfig.model_validate({"class_balancing": {"mode": "fixed", "fixed_pos_weight": 2.0}})
    assert cfg.class_balancing.fixed_pos_weight == 2.0


def test_base_override_composition(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("loop:\n  epochs: 30\n  seed: 42\n")
    override = tmp_path / "experiment.yaml"
    override.write_text("_base_: base.yaml\nloop:\n  epochs: 5\n  seed: 42\n")
    cfg = load_config(override, TrainConfig)
    assert cfg.loop.epochs == 5
