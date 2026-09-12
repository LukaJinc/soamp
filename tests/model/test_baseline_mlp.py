import pytest
import torch

from soamp.model.baseline_mlp import BaselineClassifier, OrganismEncoder, OrganismEncoderError


def _model():
    return BaselineClassifier(
        peptide_feature_dim=13, organism_vocab_size=4, organism_embed_dim=8, hidden_dims=[32, 16]
    )


def _vector_model(organism_feature_dim=32):
    return BaselineClassifier(
        peptide_feature_dim=13, organism_embed_dim=8, hidden_dims=[32, 16],
        organism_output_kind="vector", organism_feature_dim=organism_feature_dim,
    )


def test_forward_output_shape_matches_batch_size():
    model = _model()
    peptide_features = torch.randn(5, 13)
    organism_idx = torch.randint(0, 4, (5,))
    out = model(peptide_features, organism_idx)
    assert out.shape == (5,)


def test_forward_no_nans_on_synthetic_batch():
    model = _model()
    peptide_features = torch.randn(8, 13)
    organism_idx = torch.randint(0, 4, (8,))
    out = model(peptide_features, organism_idx)
    assert not torch.isnan(out).any()


def test_backward_populates_gradients():
    model = _model()
    peptide_features = torch.randn(4, 13)
    organism_idx = torch.randint(0, 4, (4,))
    labels = torch.randint(0, 2, (4,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(peptide_features, organism_idx), labels
    )
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"


def test_organism_embedding_lookup_differs_per_index():
    model = _model()
    model.eval()
    peptide_features = torch.zeros(1, 13)
    out_org1 = model(peptide_features, torch.tensor([1]))
    out_org2 = model(peptide_features, torch.tensor([2]))
    assert not torch.allclose(out_org1, out_org2)


def test_default_hidden_dims_construct_expected_layer_count():
    model = BaselineClassifier(peptide_feature_dim=13, organism_vocab_size=4)
    linear_layers = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
    # hidden_dims default [32, 16] -> 3 Linear layers: in->32, 32->16, 16->1
    assert len(linear_layers) == 3


def test_organism_encoder_defaults_to_index_kind_embedding():
    model = _model()
    assert model.organism_encoder.output_kind == "index"
    assert isinstance(model.organism_encoder.encoder, torch.nn.Embedding)


def test_vector_kind_forward_output_shape_matches_batch_size():
    model = _vector_model()
    peptide_features = torch.randn(5, 13)
    organism_vec = torch.randn(5, 32)
    out = model(peptide_features, organism_vec)
    assert out.shape == (5,)


def test_vector_kind_forward_no_nans_on_synthetic_batch():
    model = _vector_model()
    peptide_features = torch.randn(8, 13)
    organism_vec = torch.randn(8, 32)
    out = model(peptide_features, organism_vec)
    assert not torch.isnan(out).any()


def test_vector_kind_backward_populates_gradients_including_projection():
    model = _vector_model()
    peptide_features = torch.randn(4, 13)
    organism_vec = torch.randn(4, 32)
    labels = torch.randint(0, 2, (4,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(peptide_features, organism_vec), labels
    )
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
    assert isinstance(model.organism_encoder.encoder, torch.nn.Linear)


def test_vector_kind_encoder_is_linear_projection_to_embed_dim():
    model = _vector_model(organism_feature_dim=32)
    assert isinstance(model.organism_encoder.encoder, torch.nn.Linear)
    assert model.organism_encoder.encoder.in_features == 32
    assert model.organism_encoder.encoder.out_features == 8


def test_organism_encoder_raises_when_index_kind_missing_vocab_size():
    with pytest.raises(OrganismEncoderError):
        OrganismEncoder(8, output_kind="index")


def test_organism_encoder_raises_when_vector_kind_missing_feature_dim():
    with pytest.raises(OrganismEncoderError):
        OrganismEncoder(8, output_kind="vector")


def test_organism_encoder_raises_on_unknown_output_kind():
    with pytest.raises(OrganismEncoderError):
        OrganismEncoder(8, output_kind="bogus", organism_vocab_size=4)
