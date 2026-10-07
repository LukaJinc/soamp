import pytest

from soamp.splitting.graph import (
    GraphBuildError,
    build_union_graph,
    compute_fingerprint_edges,
    compute_qmap_edges,
    is_qmap_scoreable,
    union_edges,
)


def test_is_qmap_scoreable():
    assert is_qmap_scoreable("AAAAAAAAAA") is True
    assert is_qmap_scoreable("AAAAAAAAAa") is True  # lowercase (D-form) is still scoreable
    assert is_qmap_scoreable("AAAXAAAAAA") is False
    assert is_qmap_scoreable("AAAxAAAAAA") is False


def test_fingerprint_edges_identical_smiles_get_near_1_similarity():
    ids = ["1", "2"]
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]  # aspirin, twice
    edges = compute_fingerprint_edges(ids, smiles, threshold=0.5)
    assert edges[(0, 1)] == pytest.approx(1.0)


def test_fingerprint_edges_unrelated_smiles_get_no_edge():
    ids = ["1", "2"]
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCCCCCCCCCC"]  # aspirin vs hexadecane
    edges = compute_fingerprint_edges(ids, smiles, threshold=0.5)
    assert edges == {}


def test_fingerprint_edges_unparseable_smiles_contributes_no_edges():
    ids = ["1", "2", "3"]
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "not a smiles", "CC(=O)Oc1ccccc1C(=O)O"]
    edges = compute_fingerprint_edges(ids, smiles, threshold=0.5)
    assert all(1 not in edge for edge in edges)  # index 1 (unparseable) never appears
    assert edges.get((0, 2)) == pytest.approx(1.0)  # identical aspirin pair still connects


def test_fingerprint_edges_raises_on_length_mismatch():
    with pytest.raises(GraphBuildError):
        compute_fingerprint_edges(["1", "2"], ["CCO"], threshold=0.5)


# Single-substitution near-identical pairs, mirroring tests/data/test_splitting.py's
# _PAIRS convention.
def test_qmap_edges_near_identical_sequences_get_an_edge():
    ids = ["1", "2"]
    seqs = ["AAAAAAAAAA", "AAAAAAAAAC"]
    edges = compute_qmap_edges(ids, seqs, threshold=0.6, show_progress=False)
    assert (0, 1) in edges
    assert edges[(0, 1)] >= 0.6


def test_qmap_edges_x_placeholder_peptide_gets_no_edges():
    ids = ["1", "2"]
    seqs = ["AAAAAAAAAA", "AAAAAAAAXA"]  # "2" is unscoreable
    edges = compute_qmap_edges(ids, seqs, threshold=0.1, show_progress=False)
    assert edges == {}


def test_qmap_edges_x_placeholder_excluded_but_others_still_score():
    ids = ["1", "2", "3"]
    seqs = ["AAAAAAAAAA", "AAAAAAAAXA", "AAAAAAAAAC"]
    edges = compute_qmap_edges(ids, seqs, threshold=0.6, show_progress=False)
    assert (0, 2) in edges  # "1" and "3" both scoreable and near-identical
    assert all(1 not in edge for edge in edges)  # "2" (X placeholder) contributes nothing


def test_qmap_edges_raises_on_length_mismatch():
    with pytest.raises(GraphBuildError):
        compute_qmap_edges(["1", "2"], ["AAAAAAAAAA"], threshold=0.6)


def test_union_edges_dedupes_and_combines():
    fp_edges = {(0, 1): 0.9, (1, 2): 0.7}
    qm_edges = {(0, 1): 0.65, (2, 3): 0.6}
    result = union_edges(fp_edges, qm_edges)
    assert set(result) == {(0, 1), (1, 2), (2, 3)}


def test_build_union_graph_keeps_isolated_nodes():
    fp_edges = {(0, 1): 0.9}
    graph = build_union_graph(4, fp_edges, {})
    assert graph.vcount() == 4
    assert graph.ecount() == 1


def test_duplicate_pairs_links_identical_smiles_and_non_x_sequences_only():
    from soamp.splitting.graph import duplicate_pairs

    sequences = ["AAAA", "aaaa", "AXAA", "AXAA", "KKKK"]
    smiles = ["C1", "C2", "C3", "C4", "C1"]
    # (0,1) same sequence case-folded; (0,4) same SMILES; X-sequences 2,3 are
    # NOT linked by sequence (different non-canonical residues, different SMILES).
    assert duplicate_pairs(sequences, smiles) == [(0, 1), (0, 4)]


def test_contract_hard_links_groups_linked_nodes_and_drops_self_loops():
    from soamp.splitting.graph import contract_hard_links

    graph, node_of = contract_hard_links(5, [(0, 1), (1, 2)], {(0, 1): 1.0, (2, 3): 0.7, (3, 4): 0.7})
    assert node_of[0] == node_of[1] == node_of[2]
    assert len({node_of[0], node_of[3], node_of[4]}) == 3
    assert graph.vcount() == 3 and graph.ecount() == 2
    assert not any(a == b for a, b in graph.get_edgelist())
