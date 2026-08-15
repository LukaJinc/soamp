import pytest

from soamp.features.organism import OrganismVocabError, build_vocab, encode


def test_build_vocab_assigns_indices_skipping_unknown():
    vocab = build_vocab(["Escherichia coli", "Staphylococcus aureus"], unknown_index=0)
    assert 0 not in vocab.values()
    assert sorted(vocab.values()) == [1, 2]


def test_build_vocab_is_deterministic_given_input_order():
    v1 = build_vocab(["b", "a", "c"])
    v2 = build_vocab(["c", "b", "a"])
    assert v1 == v2


def test_build_vocab_dedups_repeated_organisms():
    vocab = build_vocab(["Escherichia coli", "Escherichia coli", "Staphylococcus aureus"])
    assert len(vocab) == 2


def test_build_vocab_raises_on_empty_list():
    with pytest.raises(OrganismVocabError):
        build_vocab([])


def test_build_vocab_raises_on_blank_organism():
    with pytest.raises(OrganismVocabError):
        build_vocab(["Escherichia coli", "   "])


def test_encode_known_organism():
    vocab = build_vocab(["Escherichia coli", "Staphylococcus aureus"])
    assert encode(vocab, "Escherichia coli") == vocab["Escherichia coli"]


def test_encode_unknown_organism_returns_reserved_index():
    vocab = build_vocab(["Escherichia coli"], unknown_index=0)
    assert encode(vocab, "Some New Organism", unknown_index=0) == 0
