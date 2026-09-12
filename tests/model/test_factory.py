import torch
import pytest

from soamp.data.factory import DatasetBundle, Featurization
from soamp.model.attention_fusion import AttentionFusionClassifier
from soamp.model.factory import ModelFactoryError, build_model


def _bundle(**overrides):
    defaults = dict(
        descriptor_names=["d0", "d1", "d2", "d3", "d4"],
        peptide_feature_dim=5,
        scaler={"descriptor_names": [], "mean": [], "scale": []},
        organism_vocab={"Escherichia coli": 1, "Staphylococcus aureus": 2},
        unknown_index=0,
        organism_vocab_size=3,
    )
    defaults.update(overrides)
    featurization = Featurization(**defaults)
    return DatasetBundle(datasets={}, featurization=featurization)


def _first_linear_layer(model):
    return next(m for m in model.mlp if isinstance(m, torch.nn.Linear))


def test_build_model_from_explicit_dims():
    model = build_model(peptide_feature_dim=13, organism_vocab_size=4)
    assert model.organism_encoder.encoder.num_embeddings == 4
    assert _first_linear_layer(model).in_features == 13 + 8  # default organism_embed_dim


def test_build_model_from_dataset_bundle_dims():
    bundle = _bundle()
    model = build_model(bundle, hidden_dims=[8])
    assert model.organism_encoder.encoder.num_embeddings == bundle.featurization.organism_vocab_size
    assert _first_linear_layer(model).in_features == bundle.featurization.peptide_feature_dim + 8


def test_build_model_raises_without_bundle_or_dims():
    with pytest.raises(ModelFactoryError):
        build_model()


def test_build_model_raises_with_only_one_explicit_dim():
    with pytest.raises(ModelFactoryError):
        build_model(peptide_feature_dim=13)


def test_build_model_vector_organism_kind_from_explicit_dims():
    model = build_model(
        peptide_feature_dim=13, organism_output_kind="vector", organism_feature_dim=32
    )
    assert isinstance(model.organism_encoder.encoder, torch.nn.Linear)
    assert model.organism_encoder.encoder.in_features == 32
    assert _first_linear_layer(model).in_features == 13 + 8


def test_build_model_vector_organism_kind_from_dataset_bundle():
    bundle = _bundle(
        organism_vocab=None, unknown_index=0, organism_vocab_size=None,
        organism_output_kind="vector", organism_feature_dim=32,
    )
    model = build_model(bundle)
    assert isinstance(model.organism_encoder.encoder, torch.nn.Linear)
    assert model.organism_encoder.encoder.in_features == 32


def test_build_model_raises_when_vector_kind_missing_feature_dim():
    with pytest.raises(ModelFactoryError):
        build_model(peptide_feature_dim=13, organism_output_kind="vector")


def test_build_model_attention_fusion_from_explicit_dims():
    model = build_model(
        architecture="attention_fusion_classifier",
        peptide_feature_dim=13, organism_vocab_size=4, projection_dim=16,
    )
    assert isinstance(model, AttentionFusionClassifier)
    assert model.peptide_projection[0].in_features == 13
    assert model.peptide_projection[0].out_features == 16


def test_build_model_attention_fusion_from_dataset_bundle():
    bundle = _bundle()
    model = build_model(bundle, architecture="attention_fusion_classifier", projection_dim=16)
    assert isinstance(model, AttentionFusionClassifier)
    assert model.organism_encoder.encoder.num_embeddings == bundle.featurization.organism_vocab_size


def test_build_model_raises_on_unknown_architecture():
    with pytest.raises(ModelFactoryError):
        build_model(peptide_feature_dim=13, organism_vocab_size=4, architecture="not_a_real_architecture")
