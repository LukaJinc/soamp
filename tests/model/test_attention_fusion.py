import torch

from soamp.model.attention_fusion import AttentionFusionClassifier


def _model(**overrides):
    defaults = dict(
        peptide_feature_dim=13, organism_vocab_size=4,
        projection_dim=16, num_attention_heads=2, hidden_dims=[8],
    )
    defaults.update(overrides)
    return AttentionFusionClassifier(**defaults)


def _vector_model(organism_feature_dim=32, **overrides):
    defaults = dict(
        peptide_feature_dim=13, projection_dim=16, num_attention_heads=2, hidden_dims=[8],
        organism_output_kind="vector", organism_feature_dim=organism_feature_dim,
    )
    defaults.update(overrides)
    return AttentionFusionClassifier(**defaults)


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


def test_backward_populates_all_gradients():
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


def test_vector_kind_forward_output_shape_and_no_nans():
    model = _vector_model()
    peptide_features = torch.randn(5, 13)
    organism_vec = torch.randn(5, 32)
    out = model(peptide_features, organism_vec)
    assert out.shape == (5,)
    assert not torch.isnan(out).any()


def test_vector_kind_backward_populates_gradients():
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


def test_num_attention_layers_controls_module_list_length():
    model = _model(num_attention_layers=3)
    assert len(model.attention_layers) == 3
    assert len(model.attention_norms) == 3


def test_default_hidden_dims_construct_expected_layer_count():
    model = AttentionFusionClassifier(peptide_feature_dim=13, organism_vocab_size=4)
    linear_layers = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
    # default hidden_dims=[64, 32] -> 3 Linear layers: in->64, 64->32, 32->1
    assert len(linear_layers) == 3


def test_changing_organism_input_changes_output():
    """Sanity check that organism info actually reaches the output through
    attention -- would have caught a "concat then attend on 1 token" bug
    where attention is a no-op and organism content might still leak through
    only via the (untested) encoder, not via any real cross-token mixing."""
    model = _model()
    model.eval()
    peptide_features = torch.randn(1, 13)
    with torch.no_grad():
        out_org1 = model(peptide_features, torch.tensor([1]))
        out_org2 = model(peptide_features, torch.tensor([2]))
    assert not torch.allclose(out_org1, out_org2)


def test_slot_embedding_participates_in_output():
    """Zeroing the slot embedding should change the output vs. the
    initialized (nonzero) version -- confirms the slot embedding is
    actually wired into the forward pass, not a dead parameter."""
    model = _model()
    model.eval()
    peptide_features = torch.randn(2, 13)
    organism_idx = torch.tensor([1, 2])
    with torch.no_grad():
        out_before = model(peptide_features, organism_idx).clone()
        model.slot_embedding.weight.zero_()
        out_after = model(peptide_features, organism_idx)
    assert not torch.allclose(out_before, out_after)
