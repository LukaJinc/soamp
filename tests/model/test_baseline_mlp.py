import torch

from soamp.model.baseline_mlp import BaselineClassifier


def _model():
    return BaselineClassifier(
        peptide_feature_dim=13, organism_vocab_size=4, organism_embed_dim=8, hidden_dims=[32, 16]
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
