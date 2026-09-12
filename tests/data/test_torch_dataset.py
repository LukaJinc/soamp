import pytest
import torch

from soamp.data.torch_dataset import PeptideOrganismDataset, PeptideOrganismDatasetError
from soamp.features.organism_featurizers import VocabEmbeddingOrganismFeaturizer


class _FakeVectorOrganismFeaturizer:
    """Test double for a vector-based organism strategy (no concrete
    implementation ships yet -- see soamp.features.organism_featurizers)."""

    output_kind = "vector"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def fit(self, fit_organisms):
        pass

    def encode(self, organism):
        return self.vectors[organism]


def _organism_featurizer():
    return VocabEmbeddingOrganismFeaturizer(
        unknown_index=0, vocab={"Escherichia coli": 1, "Staphylococcus aureus": 2}
    )


def _dataset(rows=None, organism_featurizer=None):
    rows = rows or [
        {"peptide_id": "1", "organism": "Escherichia coli", "label": "active"},
        {"peptide_id": "2", "organism": "Staphylococcus aureus", "label": "inactive"},
    ]
    peptide_features = {
        "1": {"a": 1.0, "b": 10.0},
        "2": {"a": 3.0, "b": 30.0},
    }
    return PeptideOrganismDataset(
        rows=rows,
        peptide_features=peptide_features,
        descriptor_names=["a", "b"],
        scaler_mean=[2.0, 20.0],
        scaler_scale=[1.0, 10.0],
        organism_featurizer=organism_featurizer or _organism_featurizer(),
    )


def test_len_matches_row_count():
    assert len(_dataset()) == 2


def test_getitem_returns_expected_shapes_and_dtypes():
    ds = _dataset()
    features, organism_idx, label = ds[0]
    assert features.shape == (2,)
    assert features.dtype == torch.float32
    assert organism_idx.shape == ()
    assert organism_idx.dtype == torch.long
    assert label.shape == ()
    assert label.dtype == torch.float32


def test_getitem_applies_scaling_correctly():
    ds = _dataset()
    features, _, _ = ds[0]
    # peptide "1": a=1.0 -> (1-2)/1 = -1.0; b=10.0 -> (10-20)/10 = -1.0
    assert torch.allclose(features, torch.tensor([-1.0, -1.0]))


def test_getitem_unknown_organism_maps_to_index_0():
    rows = [{"peptide_id": "1", "organism": "Novel Organism", "label": "active"}]
    ds = _dataset(rows)
    _, organism_idx, _ = ds[0]
    assert organism_idx.item() == 0


def test_getitem_raises_when_peptide_features_missing():
    rows = [{"peptide_id": "999", "organism": "Escherichia coli", "label": "active"}]
    ds = _dataset(rows)
    with pytest.raises(PeptideOrganismDatasetError):
        ds[0]


def test_label_encoding_active_is_1_inactive_is_0():
    ds = _dataset()
    _, _, label0 = ds[0]
    _, _, label1 = ds[1]
    assert label0.item() == 1.0
    assert label1.item() == 0.0


def test_init_raises_on_length_mismatch():
    with pytest.raises(PeptideOrganismDatasetError):
        PeptideOrganismDataset(
            rows=[],
            peptide_features={},
            descriptor_names=["a", "b"],
            scaler_mean=[1.0],
            scaler_scale=[1.0, 1.0],
            organism_featurizer=_organism_featurizer(),
        )


def test_getitem_vector_organism_featurizer_returns_float_tensor():
    rows = [{"peptide_id": "1", "organism": "Escherichia coli", "label": "active"}]
    featurizer = _FakeVectorOrganismFeaturizer({"Escherichia coli": [1.0, 2.0]})
    ds = _dataset(rows, organism_featurizer=featurizer)
    _, organism_value, _ = ds[0]
    assert organism_value.shape == (2,)
    assert organism_value.dtype == torch.float32
    assert torch.allclose(organism_value, torch.tensor([1.0, 2.0]))
