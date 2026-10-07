"""Builds the fingerprint+QMAP union similarity graph.

Two independent near-duplicate detectors, unioned: Morgan/ECFP fingerprint
Tanimoto similarity (computed from `smiles`, 100% peptide coverage) and QMAP
BLOSUM45 global sequence-identity (computed from `sequence`, only covers
peptides without DBAASP's `X`/`x` unresolved-residue placeholder). An edge
exists between two peptides if EITHER method considers them near-duplicates
-- a peptide QMAP can't score simply contributes no QMAP edges and relies
entirely on its fingerprint edges, which is expected, not a gap to fix (see
the session that decided this methodology).

The other two clustering methods explored alongside these two (RDKit-
descriptor and PeptideCLM k-means) were deliberately dropped: they had
near-zero pairwise agreement (ARI) with everything else and weak silhouette
scores, meaning they measure coarse property-space region, not near-duplicate
similarity -- not useful for leakage prevention.
"""
import igraph as ig
from qmap.toolkit.aligner import create_edgelist
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class GraphBuildError(ValueError):
    """Raised on inconsistent peptide_ids/sequences/smiles inputs (mismatched
    lengths) passed into the edge builders."""


def is_qmap_scoreable(sequence: str) -> bool:
    """True iff sequence contains no 'X'/'x' unresolved-residue placeholder.
    Matches scripts/EDA/03_qmap_sequence_clustering.ipynb's
    classify_qmap_input."""
    return "X" not in sequence and "x" not in sequence


def compute_fingerprint_edges(
    peptide_ids: list[str],
    smiles: list[str],
    threshold: float,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[tuple[int, int], float]:
    """Returns {(i, j): tanimoto_similarity} for every i<j (positional index
    into peptide_ids/smiles) with Morgan/ECFP Tanimoto similarity >=
    threshold. A SMILES that fails Chem.MolFromSmiles parsing contributes
    zero edges for that index -- a null vote, not an error (mirrors
    scripts/EDA/01_fingerprint_and_descriptor_clustering.ipynb's handling)."""
    if len(peptide_ids) != len(smiles):
        raise GraphBuildError(
            f"peptide_ids ({len(peptide_ids)}) and smiles ({len(smiles)}) "
            f"must be the same length"
        )

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
    fps = [morgan_gen.GetFingerprint(m) if m is not None else None for m in mols]
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_idx]

    edges: dict[tuple[int, int], float] = {}
    for local_i in range(1, len(valid_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[local_i], valid_fps[:local_i])
        for local_j, sim in enumerate(sims):
            if sim >= threshold:
                gi, gj = valid_idx[local_j], valid_idx[local_i]
                edges[(min(gi, gj), max(gi, gj))] = sim
    return edges


def compute_qmap_edges(
    peptide_ids: list[str],
    sequences: list[str],
    threshold: float,
    *,
    matrix: str = "blosum45",
    gap_open: int = 5,
    gap_extension: int = 1,
    use_cache: bool = True,
    show_progress: bool = True,
    num_threads: int | None = None,
) -> dict[tuple[int, int], float]:
    """Returns {(i, j): identity} for every i<j (positional index into
    peptide_ids/sequences) with BLOSUM45 global-alignment identity >=
    threshold, wrapping qmap.toolkit.aligner.create_edgelist. Peptides
    failing is_qmap_scoreable are excluded from the alignment input
    entirely -- zero QMAP-edge contribution for their index, not an error.
    Remaining sequences are .upper()'d (case-fold linearization: DBAASP
    encodes D-form residues in lowercase). Returned edges are keyed by
    index into the ORIGINAL peptide_ids/sequences array, so the result
    unions directly against compute_fingerprint_edges's output with no
    remapping at the call site."""
    if len(peptide_ids) != len(sequences):
        raise GraphBuildError(
            f"peptide_ids ({len(peptide_ids)}) and sequences ({len(sequences)}) "
            f"must be the same length"
        )

    scoreable_positions = [i for i, seq in enumerate(sequences) if is_qmap_scoreable(seq)]
    if len(scoreable_positions) < 2:
        return {}
    scoreable_sequences = [sequences[i].upper() for i in scoreable_positions]

    local_edges = create_edgelist(
        scoreable_sequences,
        threshold=threshold,
        matrix=matrix,
        gap_open=gap_open,
        gap_extension=gap_extension,
        use_cache=use_cache,
        show_progress=show_progress,
        num_threads=num_threads,
    )
    edges: dict[tuple[int, int], float] = {}
    for (i, j), identity in local_edges.items():
        gi, gj = scoreable_positions[i], scoreable_positions[j]
        edges[(min(gi, gj), max(gi, gj))] = identity
    return edges


def union_edges(*edge_sets: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    """Deduplicated union of edge-key sets (weights discarded -- Leiden runs
    unweighted here, matching every leiden_community_detection call site in
    this repo). Keys are already (min, max)-normalized by the two builders
    above, so a plain union of .keys() is sufficient."""
    seen: set[tuple[int, int]] = set()
    for edges in edge_sets:
        seen |= edges.keys()
    return sorted(seen)


def build_union_graph(n_nodes: int, *edge_sets: dict[tuple[int, int], float]) -> ig.Graph:
    """igraph.Graph(n=n_nodes, edges=union_edges(*edge_sets), directed=False)
    -- same construction shape as qmap.toolkit.clustering.build_graph.build_graph.
    A node with edges in neither input set is still present as an isolated
    vertex, never dropped."""
    return ig.Graph(n=n_nodes, edges=union_edges(*edge_sets), directed=False)


def duplicate_pairs(sequences: list[str], smiles: list[str]) -> list[tuple[int, int]]:
    """Index pairs (i<j) of peptides that are the same molecule by label:
    identical SMILES, or identical case-folded sequence among QMAP-scoreable
    ones. Sequences containing the 'X' placeholder are deliberately skipped:
    'X' stands for different non-canonical residues in different peptides, so
    the string alone does not identify the molecule (SMILES does)."""
    if len(sequences) != len(smiles):
        raise GraphBuildError(
            f"sequences ({len(sequences)}) and smiles ({len(smiles)}) must be the same length"
        )
    pairs: set[tuple[int, int]] = set()
    for keys in (
        [smi or None for smi in smiles],
        [seq.upper() if is_qmap_scoreable(seq) else None for seq in sequences],
    ):
        first_seen: dict[str, int] = {}
        for idx, key in enumerate(keys):
            if key is None:
                continue
            if key in first_seen:
                pairs.add((first_seen[key], idx))
            else:
                first_seen[key] = idx
    return sorted(pairs)


def contract_hard_links(
    n_nodes: int, linked_pairs: list[tuple[int, int]], *edge_sets: dict[tuple[int, int], float]
) -> tuple[ig.Graph, list[int]]:
    """Contracts every connected group of `linked_pairs` (duplicate /
    near-duplicate peptides) into a single node and returns
    (contracted union graph, node_of_peptide). Clustering the contracted
    graph guarantees linked peptides share a community without the giant
    cluster that merging already-formed communities produces (communities
    chain together through many cross-community links). The contracted graph
    is simple: parallel edges collapse and self-loops are dropped."""
    hard = ig.Graph(n=n_nodes, edges=linked_pairs, directed=False)
    node_of_peptide = hard.connected_components().membership
    n_groups = max(node_of_peptide) + 1 if node_of_peptide else 0
    edges = {
        (min(a, b), max(a, b))
        for (i, j) in union_edges(*edge_sets)
        if (a := node_of_peptide[i]) != (b := node_of_peptide[j])
    }
    return ig.Graph(n=n_groups, edges=sorted(edges), directed=False), node_of_peptide
