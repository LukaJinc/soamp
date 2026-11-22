import pytest

from soamp.splitting.alignment import (
    TokenAlphabet,
    compute_edges,
    contract_hard_links,
    global_identity,
    make_aligner,
)

_ALPHABET = TokenAlphabet(["mod:Orn", "mod:DAB"])


def _identity(a, b):
    aligner = make_aligner(_ALPHABET)
    return global_identity(aligner, _ALPHABET.encode(a), _ALPHABET.encode(b))


def test_identical_sequences():
    assert _identity(list("KWKLFK"), list("KWKLFK")) == 1.0


def test_single_substitution_matches_qmap_value():
    # QMAP gives 15/16 for one substitution in 16 residues
    seq = list("KWKLFKKIGAVLKVLA")
    other = seq.copy()
    other[7] = "A"
    assert _identity(seq, other) == pytest.approx(15 / 16)


def test_same_unusual_residue_aligns_different_does_not():
    base = list("KWKLFKK")
    with_orn = base[:3] + ["mod:Orn"] + base[4:]
    with_dab = base[:3] + ["mod:DAB"] + base[4:]
    assert _identity(with_orn, with_orn) == 1.0
    assert _identity(with_orn, with_dab) == pytest.approx(6 / 7)  # QMAP could not tell these apart


def test_modification_chars_are_distinct_and_private():
    chars = {_ALPHABET.encode(["mod:Orn"]), _ALPHABET.encode(["mod:DAB"])}
    assert len(chars) == 2
    assert not chars & set("ACDEFGHIKLMNPQRSTVWY")


def test_rejects_non_modification_tokens():
    with pytest.raises(ValueError):
        TokenAlphabet(["Orn"])


def test_compute_edges_threshold_and_parallel_agree():
    seqs = [
        _ALPHABET.encode(list(s))
        for s in ["KWKLFKKIGAVLKVLA", "KWKLFKKIGAVLKVLG", "GGGGSSSSPPPPDDDD", "KWKLFKKIGAVLKVLA"]
    ]
    serial = compute_edges(seqs, _ALPHABET, 0.9, n_jobs=1)
    assert set(serial) == {(0, 1), (0, 3), (1, 3)}
    assert serial[(0, 3)] == 1.0
    assert compute_edges(seqs, _ALPHABET, 0.9, n_jobs=2) == serial


def test_contract_hard_links_merges_only_pairs_at_or_above_threshold():
    edges = {(0, 1): 0.95, (1, 2): 0.7, (2, 3): 0.92, (0, 3): 0.65}
    group, contracted = contract_hard_links(5, edges, 0.9)
    assert group[0] == group[1] and group[2] == group[3] and group[0] != group[2]
    assert group[4] not in (group[0], group[2])          # isolated node stays alone
    a, b = int(group[0]), int(group[2])
    # every original edge is kept: 2 inside merged nodes (self-loops), 2 between them (parallel)
    assert sorted(contracted) == sorted([(a, a), (b, b), (a, b), (a, b)])


def test_contract_hard_links_chains_are_transitive():
    group, contracted = contract_hard_links(3, {(0, 1): 0.95, (1, 2): 0.95}, 0.9)
    assert len(set(group)) == 1 and contracted == [(int(group[0]),) * 2] * 2
