import pytest

from soamp.features.peptide import DESCRIPTOR_NAMES, build_peptide_feature_rows
from soamp.features.peptide_featurizers import (
    PeptideCLMFeaturizer,
    PeptideFeaturizerError,
    RDKitDescriptorFeaturizer,
    build_peptide_featurizer,
)

GLYCINE = "C(C(=O)O)N"
ACETIC_ACID = "CC(=O)O"

_UNIQUE_PEPTIDES = [
    {"peptide_id": "1", "smiles": GLYCINE},
    {"peptide_id": "2", "smiles": ACETIC_ACID},
]


def test_rdkit_featurizer_default_feature_names_match_descriptor_names():
    featurizer = RDKitDescriptorFeaturizer()
    assert featurizer.feature_names == list(DESCRIPTOR_NAMES)


def test_rdkit_featurizer_transform_matches_build_peptide_feature_rows():
    featurizer = RDKitDescriptorFeaturizer()
    featurizer.fit(_UNIQUE_PEPTIDES)  # no-op, must not raise
    assert featurizer.transform(_UNIQUE_PEPTIDES) == build_peptide_feature_rows(_UNIQUE_PEPTIDES)


def test_rdkit_featurizer_respects_custom_descriptor_names():
    featurizer = RDKitDescriptorFeaturizer(descriptor_names=["MolWt", "TPSA"])
    rows = featurizer.transform(_UNIQUE_PEPTIDES)
    assert set(rows[0].keys()) == {"peptide_id", "MolWt", "TPSA"}


def test_peptideclm_featurizer_does_not_load_model_on_construction():
    featurizer = PeptideCLMFeaturizer()
    assert featurizer._model is None
    assert featurizer._tokenizer is None


def test_peptideclm_featurizer_feature_names_are_named_dims():
    featurizer = PeptideCLMFeaturizer()
    assert featurizer.feature_names == [f"dim_{i}" for i in range(768)]
    assert len(featurizer.feature_names) == PeptideCLMFeaturizer.HIDDEN_SIZE


def test_peptideclm_featurizer_fit_is_a_noop():
    featurizer = PeptideCLMFeaturizer()
    featurizer.fit(_UNIQUE_PEPTIDES)  # must not raise or load the model
    assert featurizer._model is None


def test_build_peptide_featurizer_dispatches_rdkit():
    featurizer = build_peptide_featurizer("rdkit_descriptors")
    assert isinstance(featurizer, RDKitDescriptorFeaturizer)


def test_build_peptide_featurizer_dispatches_peptideclm():
    featurizer = build_peptide_featurizer("peptideclm_embedding")
    assert isinstance(featurizer, PeptideCLMFeaturizer)


def test_build_peptide_featurizer_passes_kwargs_through():
    featurizer = build_peptide_featurizer("rdkit_descriptors", descriptor_names=["MolWt"])
    assert featurizer.feature_names == ["MolWt"]


def test_build_peptide_featurizer_raises_on_unknown_method():
    with pytest.raises(PeptideFeaturizerError):
        build_peptide_featurizer("not_a_real_method")
