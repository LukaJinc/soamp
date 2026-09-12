import pytest
from pydantic import ValidationError

from soamp.engine.config import TrainConfig
from soamp.utils.config import load_config


def test_base_yaml_loads():
    cfg = load_config("config/train/base.yaml", TrainConfig)
    assert cfg.exp_id == "baseline_mlp_v1"
    assert cfg.class_balancing.mode == "auto"
    assert cfg.paths.checkpoint_dir.is_absolute()
    assert cfg.model.architecture == "baseline_classifier"
    assert cfg.model.baseline_classifier.hidden_dims == [32, 16]
    assert cfg.model.attention_fusion_classifier.projection_dim == 128


def test_model_active_kwargs_returns_baseline_classifier_fields_by_default():
    cfg = load_config("config/train/base.yaml", TrainConfig)
    assert cfg.model.active_kwargs() == {"organism_embed_dim": 8, "hidden_dims": [32, 16]}


def test_model_active_kwargs_returns_attention_fusion_fields_when_selected():
    cfg = TrainConfig.model_validate({"model": {"architecture": "attention_fusion_classifier"}})
    assert cfg.model.active_kwargs() == {
        "projection_dim": 128, "num_attention_heads": 4,
        "num_attention_layers": 1, "hidden_dims": [64, 32],
    }


def test_model_unknown_architecture_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"model": {"architecture": "not_a_real_architecture"}})


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


def test_loop_device_defaults_to_auto():
    cfg = load_config("config/train/base.yaml", TrainConfig)
    assert cfg.loop.device == "auto"


def test_best_metric_lower_is_better_direction():
    assert not TrainConfig.model_validate(
        {"checkpoint": {"best_metric": "val_auroc"}}
    ).checkpoint.lower_is_better
    assert TrainConfig.model_validate(
        {"checkpoint": {"best_metric": "val_loss"}}
    ).checkpoint.lower_is_better


def test_unknown_best_metric_rejected():
    """best_metric is indexed straight into the logged epoch-metrics dict, so a
    typo must fail at load time rather than KeyError mid-training."""
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"checkpoint": {"best_metric": "val_auc"}})


def test_wandb_group_and_tags_default_to_unset():
    cfg = load_config("config/train/base.yaml", TrainConfig)
    assert cfg.wandb_group is None
    assert cfg.wandb_tags == []


@pytest.mark.parametrize(
    "overlay, exp_id, peptide_features, organism_vocab",
    [
        ("exp_rdkit_vocab", "rdkit_vocab_attnfusion",
         "peptide_features_rdkit.csv", "organism_vocab_vocab_embedding.json"),
        ("exp_rdkit_kmer", "rdkit_kmer_attnfusion",
         "peptide_features_rdkit.csv", "organism_vocab_kmer_composition.json"),
        ("exp_peptideclm_vocab", "peptideclm_vocab_attnfusion",
         "peptide_features_peptideclm.csv", "organism_vocab_vocab_embedding.json"),
        ("exp_peptideclm_kmer", "peptideclm_kmer_attnfusion",
         "peptide_features_peptideclm.csv", "organism_vocab_kmer_composition.json"),
    ],
)
def test_featurization_grid_configs(overlay, exp_id, peptide_features, organism_vocab):
    """All four grid cells resolve, point at their own feature artifacts, and
    hold the architecture fixed so the comparison isolates the representation."""
    cfg = load_config(f"config/train/{overlay}.yaml", TrainConfig)
    assert cfg.exp_id == exp_id
    assert cfg.model.architecture == "attention_fusion_classifier"
    assert cfg.input_files.peptide_features_filename == peptide_features
    assert cfg.input_files.organism_vocab_filename == organism_vocab
    assert cfg.wandb_group == "featurization_grid_v1"
