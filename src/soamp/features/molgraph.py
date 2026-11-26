"""Pure RDKit molecular-graph construction for a peptide SMILES string.

SMILES-only, like soamp.features.peptide: atoms and bonds are read directly
from the molecule, so canonical, non-canonical and cyclic peptides all go
through the same code path with no residue-letter vocabulary to extend.

A graph is plain numpy (no torch import) so featurization stays importable
and testable without any model code; batching into tensors lives in
soamp.data.graph_batch.
"""
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger

from soamp.features.peptide import PeptideFeatureError

RDLogger.DisableLog("rdApp.*")

_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]  # + "other"
_DEGREES = [0, 1, 2, 3, 4, 5]                                   # + ">5"
_CHARGES = [-1, 0, 1]                                           # + "other"
_HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
]                                                               # + "other"
_NUM_HS = [0, 1, 2, 3]                                          # + ">3"
_CHIRAL_TAGS = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]                                                               # + "other"
_BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
_BOND_STEREO = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
]                                                               # + "other"


def _one_hot(value, choices) -> list[float]:
    """len(choices)+1 slots; the last one catches anything not in choices."""
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1.0
    return out


NODE_DIM = (
    len(_ELEMENTS) + 1 + len(_DEGREES) + 1 + len(_CHARGES) + 1
    + len(_HYBRIDIZATIONS) + 1 + 1 + len(_NUM_HS) + 1 + 1 + len(_CHIRAL_TAGS) + 1
)
EDGE_DIM = len(_BOND_TYPES) + 1 + 1 + 1 + len(_BOND_STEREO) + 1


@dataclass
class MolGraph:
    node_feats: np.ndarray   # (N, NODE_DIM) float32
    edge_index: np.ndarray   # (2, E) int64, both directions of every bond
    edge_feats: np.ndarray   # (E, EDGE_DIM) float32

    @property
    def num_nodes(self) -> int:
        return self.node_feats.shape[0]


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), _ELEMENTS)
        + _one_hot(atom.GetDegree(), _DEGREES)
        + _one_hot(atom.GetFormalCharge(), _CHARGES)
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic())]
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS)
        + [float(atom.IsInRing())]
        + _one_hot(atom.GetChiralTag(), _CHIRAL_TAGS)
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), _BOND_STEREO)
    )


def smiles_to_graph(smiles: str) -> MolGraph:
    """Heavy-atom graph of `smiles`. Raises PeptideFeatureError if the SMILES
    is empty or unparseable -- upstream curation/QA already validated these,
    so a failure here means a broken source-data invariant, not bad input to
    be silently skipped."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise PeptideFeatureError(f"cannot build a molecular graph from SMILES {smiles!r}")

    node_feats = np.asarray([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    sources, targets, edge_rows = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)
        sources += [i, j]
        targets += [j, i]
        edge_rows += [features, features]

    edge_index = np.asarray([sources, targets], dtype=np.int64).reshape(2, -1)
    edge_feats = np.asarray(edge_rows, dtype=np.float32).reshape(-1, EDGE_DIM)
    return MolGraph(node_feats=node_feats, edge_index=edge_index, edge_feats=edge_feats)


# ---------------------------------------------------------------------------
# Fixed-length (parameter-free) summary of a MolGraph, for models that cannot
# take a graph (trees) or to concatenate with another vector (PeptideCLM).
# Everything is a count or a fraction read straight off the graph's own
# one-hot node/edge features, so it is a fixed function of the SMILES.
# ---------------------------------------------------------------------------

def _slot_names(prefix: str, choices: list, labels: list[str] | None = None) -> list[str]:
    names = labels if labels is not None else [str(c) for c in choices]
    return [f"{prefix}_{n}" for n in names] + [f"{prefix}_other"]


_NODE_SLOT_NAMES = (
    _slot_names("elem", _ELEMENTS)
    + _slot_names("deg", _DEGREES)
    + _slot_names("charge", _CHARGES)
    + _slot_names("hyb", _HYBRIDIZATIONS, ["sp", "sp2", "sp3"])
    + ["aromatic"]
    + _slot_names("nh", _NUM_HS)
    + ["ring"]
    + _slot_names("chiral", _CHIRAL_TAGS, ["none", "cw", "ccw"])
)
_EDGE_SLOT_NAMES = (
    _slot_names("bond", _BOND_TYPES, ["single", "double", "triple", "aromatic"])
    + ["conjugated", "ring"]
    + _slot_names("stereo", _BOND_STEREO, ["none", "z", "e"])
)
assert len(_NODE_SLOT_NAMES) == NODE_DIM and len(_EDGE_SLOT_NAMES) == EDGE_DIM

_N_ELEM = len(_ELEMENTS) + 1
# unordered element pairs over bonds (C-N, C-O, ...): the 1-hop atom-environment signal
_ELEM_PAIRS = [(i, j) for i in range(_N_ELEM) for j in range(i, _N_ELEM)]
_ELEM_LABELS = _ELEMENTS + ["other"]

POOLED_FEATURE_NAMES: list[str] = (
    ["mg_n_atoms", "mg_n_bonds"]
    + [f"mg_atom_count_{n}" for n in _NODE_SLOT_NAMES]
    + [f"mg_atom_frac_{n}" for n in _NODE_SLOT_NAMES]
    + [f"mg_bond_count_{n}" for n in _EDGE_SLOT_NAMES]
    + [f"mg_bond_frac_{n}" for n in _EDGE_SLOT_NAMES]
    + [f"mg_pair_{_ELEM_LABELS[i]}_{_ELEM_LABELS[j]}" for i, j in _ELEM_PAIRS]
)


def pooled_graph_features(graph: MolGraph) -> np.ndarray:
    """(len(POOLED_FEATURE_NAMES),) float32: atom/bond counts, per-slot atom and
    bond counts and fractions, and element-pair bond counts."""
    n_atoms = graph.num_nodes
    # edge_index stores each bond twice (both directions): halve sums to count bonds
    bond_sum = graph.edge_feats.sum(axis=0) / 2.0
    n_bonds = graph.edge_feats.shape[0] / 2.0
    atom_sum = graph.node_feats.sum(axis=0)

    elem = graph.node_feats[:, :_N_ELEM].argmax(axis=1)
    pairs = np.zeros((_N_ELEM, _N_ELEM), dtype=np.float64)
    if graph.edge_index.shape[1]:
        src, dst = graph.edge_index
        np.add.at(pairs, (elem[src], elem[dst]), 1.0)
    pairs = pairs / 2.0  # each bond visited in both directions
    sym = pairs + pairs.T - np.diag(np.diag(pairs))  # unordered counts on the upper triangle
    pair_counts = [sym[i, j] for i, j in _ELEM_PAIRS]

    return np.concatenate([
        [n_atoms, n_bonds],
        atom_sum, atom_sum / max(n_atoms, 1),
        bond_sum, bond_sum / max(n_bonds, 1.0),
        pair_counts,
    ]).astype(np.float32)
