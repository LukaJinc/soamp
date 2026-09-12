import pytest

from soamp.features.organism_featurizers import (
    KmerOrganismFeaturizer,
    OrganismFeaturizerError,
    VocabEmbeddingOrganismFeaturizer,
    build_organism_featurizer,
    organism_featurizer_from_artifact,
)


def test_vocab_embedding_output_kind_is_index():
    assert VocabEmbeddingOrganismFeaturizer().output_kind == "index"


def test_vocab_embedding_vocab_size_and_feature_dim_before_fit():
    featurizer = VocabEmbeddingOrganismFeaturizer()
    assert featurizer.vocab_size is None
    assert featurizer.feature_dim is None


def test_vocab_embedding_fit_builds_vocab_skipping_unknown_index():
    featurizer = VocabEmbeddingOrganismFeaturizer(unknown_index=0)
    featurizer.fit(["Escherichia coli", "Staphylococcus aureus", "Escherichia coli"])
    assert set(featurizer.vocab.keys()) == {"Escherichia coli", "Staphylococcus aureus"}
    assert 0 not in featurizer.vocab.values()
    assert featurizer.vocab_size == len(featurizer.vocab) + 1
    assert featurizer.feature_dim is None


def test_vocab_embedding_encode_known_and_unknown_organism():
    featurizer = VocabEmbeddingOrganismFeaturizer(unknown_index=0, vocab={"Escherichia coli": 1})
    assert featurizer.encode("Escherichia coli") == 1
    assert featurizer.encode("Novel Organism") == 0


def test_vocab_embedding_to_artifact_dict_shape():
    featurizer = VocabEmbeddingOrganismFeaturizer(unknown_index=0, vocab={"Escherichia coli": 1})
    assert featurizer.to_artifact_dict() == {
        "method": "vocab_embedding",
        "unknown_index": 0,
        "vocab_size": 2,
        "vocab": {"Escherichia coli": 1},
    }


def test_build_organism_featurizer_dispatches_vocab_embedding():
    featurizer = build_organism_featurizer("vocab_embedding")
    assert isinstance(featurizer, VocabEmbeddingOrganismFeaturizer)


def test_build_organism_featurizer_passes_kwargs_through():
    featurizer = build_organism_featurizer("vocab_embedding", unknown_index=5)
    assert featurizer.unknown_index == 5


def test_build_organism_featurizer_raises_on_unknown_method():
    with pytest.raises(OrganismFeaturizerError):
        build_organism_featurizer("not_a_real_method")


def test_organism_featurizer_from_artifact_reconstructs_vocab_embedding():
    data = {
        "method": "vocab_embedding", "unknown_index": 0,
        "vocab_size": 2, "vocab": {"Escherichia coli": 1},
    }
    featurizer = organism_featurizer_from_artifact(data)
    assert isinstance(featurizer, VocabEmbeddingOrganismFeaturizer)
    assert featurizer.encode("Escherichia coli") == 1


def test_organism_featurizer_from_artifact_raises_on_unknown_method():
    with pytest.raises(OrganismFeaturizerError):
        organism_featurizer_from_artifact({"method": "not_a_real_method"})


# --- KmerOrganismFeaturizer ---
# Constructed via the species_features/genus_features pre-population args
# throughout -- exercises the featurizer logic without any network/file I/O.


def _kmer_featurizer(**overrides):
    defaults = dict(
        k_values=(1, 2),
        species_features={"Escherichia coli": [1.0, 0.0, 0.0, 0.0, 0.1] + [0.0] * 15},
        genus_features={"Pseudomonas": [0.0, 1.0, 0.0, 0.0] + [0.0] * 16},
    )
    defaults.update(overrides)
    return KmerOrganismFeaturizer(**defaults)


def test_kmer_output_kind_is_vector():
    assert KmerOrganismFeaturizer().output_kind == "vector"


def test_kmer_feature_dim_matches_k_values():
    featurizer = KmerOrganismFeaturizer(k_values=(1, 2, 3, 4))
    assert featurizer.feature_dim == 4 + 16 + 64 + 256
    featurizer_small = KmerOrganismFeaturizer(k_values=(1, 2))
    assert featurizer_small.feature_dim == 4 + 16


def test_kmer_vocab_size_is_none():
    assert KmerOrganismFeaturizer().vocab_size is None


def test_kmer_encode_species_exact_match():
    featurizer = _kmer_featurizer()
    vector = featurizer.encode("Escherichia coli")
    assert vector == [1.0, 0.0, 0.0, 0.0, 0.1] + [0.0] * 15


def test_kmer_encode_genus_fallback():
    featurizer = _kmer_featurizer()
    vector = featurizer.encode("Pseudomonas aeruginosa")  # no species entry, genus "Pseudomonas" does
    assert vector == [0.0, 1.0, 0.0, 0.0] + [0.0] * 16


def test_kmer_encode_unresolvable_organism_returns_zero_vector():
    featurizer = _kmer_featurizer()
    vector = featurizer.encode("Totally Novel Organism")
    assert vector == [0.0] * featurizer.feature_dim


def test_kmer_fit_succeeds_when_all_fit_organisms_resolvable():
    featurizer = _kmer_featurizer()
    featurizer.fit(["Escherichia coli", "Pseudomonas aeruginosa"])  # no raise


def test_kmer_fit_raises_when_a_fit_organism_is_unresolvable():
    featurizer = _kmer_featurizer()
    with pytest.raises(OrganismFeaturizerError):
        featurizer.fit(["Escherichia coli", "Totally Novel Organism"])


def test_kmer_to_artifact_dict_shape():
    featurizer = _kmer_featurizer()
    artifact = featurizer.to_artifact_dict()
    assert artifact["method"] == "kmer_composition"
    assert artifact["k_values"] == [1, 2]
    assert artifact["feature_dim"] == 20
    assert artifact["species_features"] == {"Escherichia coli": [1.0, 0.0, 0.0, 0.0, 0.1] + [0.0] * 15}
    assert artifact["genus_features"] == {"Pseudomonas": [0.0, 1.0, 0.0, 0.0] + [0.0] * 16}


def test_kmer_artifact_round_trip_via_organism_featurizer_from_artifact():
    featurizer = _kmer_featurizer()
    reconstructed = organism_featurizer_from_artifact(featurizer.to_artifact_dict())
    assert isinstance(reconstructed, KmerOrganismFeaturizer)
    assert reconstructed.encode("Escherichia coli") == featurizer.encode("Escherichia coli")
    assert reconstructed.feature_dim == featurizer.feature_dim


def test_build_organism_featurizer_dispatches_kmer_composition():
    featurizer = build_organism_featurizer("kmer_composition", k_values=(1, 2))
    assert isinstance(featurizer, KmerOrganismFeaturizer)
    assert featurizer.k_values == (1, 2)
