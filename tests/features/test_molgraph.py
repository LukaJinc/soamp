import numpy as np
import pytest

from soamp.features.molgraph import EDGE_DIM, NODE_DIM, smiles_to_graph
from soamp.features.peptide import PeptideFeatureError

GLYCINE = "C(C(=O)O)N"            # 5 heavy atoms, 4 bonds
GLY_GLY = "NCC(=O)NCC(=O)O"       # 8 heavy atoms, 7 bonds
CYCLO_GLY3 = "O=C1CNC(=O)CNC(=O)CN1"  # cyclic tripeptide: 12 heavy atoms, 9-membered ring
L_ALA = "N[C@@H](C)C(=O)O"
D_ALA = "N[C@H](C)C(=O)O"


def test_graph_shapes_and_dimensions():
    g = smiles_to_graph(GLYCINE)
    assert g.node_feats.shape == (5, NODE_DIM)
    assert g.edge_index.shape == (2, 8)           # 4 bonds, both directions
    assert g.edge_feats.shape == (8, EDGE_DIM)
    assert g.node_feats.dtype == np.float32 and g.edge_index.dtype == np.int64


def test_edges_are_bidirectional_and_in_range():
    g = smiles_to_graph(GLY_GLY)
    pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert g.edge_index.max() < g.num_nodes


def test_cyclic_peptide_has_closing_bond_flagged_in_ring():
    g = smiles_to_graph(CYCLO_GLY3)
    assert g.num_nodes == 12
    assert g.edge_index.shape[1] == 2 * 12         # a linear chain of 12 atoms would have 11 bonds
    in_ring_column = 6                              # 4 bond types + other, conjugated, then in_ring
    assert int(g.edge_feats[:, in_ring_column].sum()) == 2 * 9   # 9-membered ring, both directions


def test_node_features_are_one_hot_blocks():
    g = smiles_to_graph(GLYCINE)
    assert (g.node_feats.sum(axis=1) >= 6).all()   # every atom sets several categorical slots
    assert set(np.unique(g.node_feats)) <= {0.0, 1.0}


def test_chirality_distinguishes_enantiomers():
    assert not np.array_equal(smiles_to_graph(L_ALA).node_feats, smiles_to_graph(D_ALA).node_feats)


@pytest.mark.parametrize("bad", ["", "not-a-smiles", "C(("])
def test_invalid_smiles_raises(bad):
    with pytest.raises(PeptideFeatureError):
        smiles_to_graph(bad)
