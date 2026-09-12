import torch

from soamp.data.factory import DatasetBundle, Featurization
from soamp.engine.tracking import build_tracker, watch_model
from soamp.model.baseline_mlp import BaselineClassifier


def _bundle(**overrides):
    defaults = dict(
        descriptor_names=["d0", "d1", "d2", "d3", "d4"],
        peptide_feature_dim=5,
        scaler={"descriptor_names": [], "mean": [], "scale": []},
        organism_vocab={"Escherichia coli": 1, "Staphylococcus aureus": 2},
        unknown_index=0,
        organism_vocab_size=3,
        peptide_method="rdkit_descriptors",
        organism_method="vocab_embedding",
    )
    defaults.update(overrides)
    featurization = Featurization(**defaults)
    return DatasetBundle(datasets={}, featurization=featurization)


def test_build_tracker_merges_hyperparams_and_featurization_into_config():
    bundle = _bundle()
    run = build_tracker(
        bundle, {"epochs": 5, "seed": 42}, project="soamp-test", mode="disabled",
    )
    config = dict(run.config)
    assert config["epochs"] == 5
    assert config["seed"] == 42
    assert config["peptide_method"] == "rdkit_descriptors"
    assert config["peptide_feature_dim"] == 5
    assert config["descriptor_names"] == ["d0", "d1", "d2", "d3", "d4"]
    assert config["organism_method"] == "vocab_embedding"
    assert config["organism_output_kind"] == "index"
    assert config["organism_vocab_size"] == 3
    assert config["organism_feature_dim"] is None
    run.finish()


def test_build_tracker_featurization_overrides_colliding_hyperparam_key():
    bundle = _bundle()
    run = build_tracker(
        bundle, {"peptide_method": "stale_hand_typed_value"},
        project="soamp-test", mode="disabled",
    )
    assert dict(run.config)["peptide_method"] == "rdkit_descriptors"
    run.finish()


def test_watch_model_does_not_raise_on_forward_and_backward():
    run = build_tracker(_bundle(), {}, project="soamp-test", mode="disabled")
    model = BaselineClassifier(peptide_feature_dim=5, organism_vocab_size=3, hidden_dims=[8])
    watch_model(model)

    peptide_features = torch.randn(4, 5)
    organism_idx = torch.randint(0, 3, (4,))
    labels = torch.randint(0, 2, (4,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(peptide_features, organism_idx), labels
    )
    loss.backward()

    run.finish()
