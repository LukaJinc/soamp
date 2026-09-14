from unittest.mock import patch

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


def _fake_uncached_transform(calls):
    """Stand-in for _transform_uncached that records which peptide_ids it
    was actually asked to compute, without loading the real model."""
    def fake(self, unique_peptides):
        calls.append([p["peptide_id"] for p in unique_peptides])
        return [
            {"peptide_id": p["peptide_id"], **{name: 0.0 for name in self.feature_names}}
            for p in unique_peptides
        ]
    return fake


def test_peptideclm_featurizer_transform_caches_by_peptide_id_within_one_instance():
    calls = []
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_uncached_transform(calls)):
        featurizer = PeptideCLMFeaturizer()
        featurizer.transform(_UNIQUE_PEPTIDES)
        featurizer.transform(_UNIQUE_PEPTIDES)  # same peptides again
    assert calls == [["1", "2"]]  # second call is a full cache hit, computes nothing


def test_peptideclm_featurizer_shares_cache_across_separate_instances_when_given_one():
    """The whole point of the injectable cache: k-fold CV constructs a fresh
    PeptideCLMFeaturizer per fold, but embeddings are frozen/deterministic --
    passing the same dict object into each fresh instance means a peptide
    seen in an earlier fold is never recomputed."""
    calls = []
    shared_cache: dict = {}
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_uncached_transform(calls)):
        first = PeptideCLMFeaturizer(cache=shared_cache)
        first.transform(_UNIQUE_PEPTIDES)  # populates "1" and "2"

        second = PeptideCLMFeaturizer(cache=shared_cache)  # fresh instance, same cache
        second.transform(_UNIQUE_PEPTIDES + [{"peptide_id": "3", "smiles": "CCO"}])

    assert calls == [["1", "2"], ["3"]]  # second call only computes the new id


def test_peptideclm_featurizer_defaults_to_a_private_cache_when_none_given():
    """Two instances with no shared cache must not leak state into each
    other -- every existing single-call caller relies on this."""
    calls = []
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_uncached_transform(calls)):
        first = PeptideCLMFeaturizer()
        first.transform(_UNIQUE_PEPTIDES)

        second = PeptideCLMFeaturizer()  # no cache= passed -- independent
        second.transform(_UNIQUE_PEPTIDES)

    assert calls == [["1", "2"], ["1", "2"]]  # second instance recomputes, no leakage


def test_peptideclm_featurizer_transform_returns_rows_in_requested_order():
    """Cache-hit rows and freshly-computed rows must interleave correctly --
    a naive implementation that appends fresh rows before returning could
    silently reorder the output relative to unique_peptides."""
    calls = []
    shared_cache: dict = {}
    with patch.object(PeptideCLMFeaturizer, "_transform_uncached", _fake_uncached_transform(calls)):
        featurizer = PeptideCLMFeaturizer(cache=shared_cache)
        featurizer.transform([_UNIQUE_PEPTIDES[0]])  # warm the cache with "1" only

        result = featurizer.transform([_UNIQUE_PEPTIDES[1], _UNIQUE_PEPTIDES[0]])  # "2" then "1"
    assert [row["peptide_id"] for row in result] == ["2", "1"]
