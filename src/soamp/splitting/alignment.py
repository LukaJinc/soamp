"""QMAP-like global-alignment identity over an extended token alphabet.

Why this exists: QMAP's compiled aligner (``qmap._pwiden_engine``) accepts only
a *named* substitution matrix and knows 24 symbols (20 amino acids + B, X, Z,
case-insensitive); every other character collapses into one "unknown" symbol
that matches every other unknown. DBAASP peptides carry hundreds of distinct
unusual residues, so they cannot be kept apart through QMAP. This module
re-implements the same kind of computation with one alphabet entry per
unusual residue.

It is an *approximation* of QMAP, not a clone (measured on 400 real
canonical-only peptides: 65% of pairs identical, correlation 0.985, edge-set
Jaccard vs QMAP 0.79 / 0.92 / 1.00 at identity thresholds 0.4 / 0.6 / 0.8;
QMAP's denominator and tie-breaking among equal-score alignments differ and
its engine is closed). Use this one definition for every peptide in a graph
so that edges are mutually consistent.

Definition used: global (Needleman-Wunsch) alignment, BLOSUM45 for canonical
pairs, gap open 5 / extend 1 (QMAP defaults); identity = matches / alignment
length. An unusual residue scores the BLOSUM45 mean diagonal against itself,
and the BLOSUM45 ``X``-``X`` / ``X``-aa scores against anything else.
"""
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import igraph as ig
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from soamp.splitting.tokens import CANONICAL, MOD_PREFIX

_FIRST_MOD_CODEPOINT = 0x100


class TokenAlphabet:
    """Maps tokens to single characters Biopython can align.

    Canonical residues keep their letter; each ``mod:<name>`` token gets its
    own private character, so two different unusual residues never match.
    """

    def __init__(self, modification_tokens: Sequence[str]):
        mods = sorted(set(modification_tokens))
        bad = [t for t in mods if not t.startswith(MOD_PREFIX)]
        if bad:
            raise ValueError(f"not modification tokens: {bad[:3]}")
        self._char = {t: chr(_FIRST_MOD_CODEPOINT + i) for i, t in enumerate(mods)}
        self._char.update({a: a for a in CANONICAL})
        self.modification_chars = "".join(self._char[t] for t in mods)
        self.chars = CANONICAL + self.modification_chars

    def encode(self, tokens: Sequence[str]) -> str:
        return "".join(self._char[t] for t in tokens)

    def token_for(self, char: str) -> str:
        return next(t for t, c in self._char.items() if c == char)


def build_substitution_matrix(alphabet: TokenAlphabet):
    """BLOSUM45 extended with one row/column per unusual residue."""
    blosum = substitution_matrices.load("BLOSUM45")
    n_canon = len(CANONICAL)
    size = len(alphabet.chars)
    data = np.zeros((size, size))
    for i, a in enumerate(CANONICAL):
        for j, b in enumerate(CANONICAL):
            data[i, j] = blosum[a, b]
    self_score = float(np.mean([blosum[a, a] for a in CANONICAL]))
    mod_vs_other = float(blosum["X", "X"])
    data[n_canon:, :] = blosum["X", "A"]  # BLOSUM45 X row is flat (0.0) for aa
    data[:, n_canon:] = blosum["X", "A"]
    data[n_canon:, n_canon:] = mod_vs_other
    for k in range(n_canon, size):
        data[k, k] = self_score
    return substitution_matrices.Array(alphabet=alphabet.chars, dims=2, data=data)


def make_aligner(alphabet: TokenAlphabet, gap_open: float = 5, gap_extend: float = 1) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = build_substitution_matrix(alphabet)
    aligner.open_gap_score = -gap_open
    aligner.extend_gap_score = -gap_extend
    return aligner


def global_identity(aligner: PairwiseAligner, a: str, b: str) -> float:
    """matches / alignment length of one optimal global alignment."""
    best = aligner.align(a, b)[0]
    matches = sum(
        a[s1 + k] == b[s2 + k]
        for (s1, e1), (s2, _) in zip(*best.aligned)
        for k in range(e1 - s1)
    )
    return matches / best.shape[1]


# ---- pairwise edge list ---------------------------------------------------

_STATE: dict = {}


def _init_worker(sequences, chars, gap_open, gap_extend, threshold):
    alphabet = TokenAlphabet([])
    alphabet.chars = chars
    alphabet.modification_chars = chars[len(CANONICAL):]
    _STATE["aligner"] = make_aligner(alphabet, gap_open, gap_extend)
    _STATE["seqs"] = sequences
    _STATE["lens"] = np.array([len(s) for s in sequences])
    _STATE["threshold"] = threshold


def _rows(rows: list[int]) -> list[tuple[int, int, float]]:
    seqs, lens = _STATE["seqs"], _STATE["lens"]
    aligner, threshold = _STATE["aligner"], _STATE["threshold"]
    out = []
    for i in rows:
        # identity <= min(len)/max(len): skip pairs that cannot reach the threshold
        js = np.arange(i + 1, len(seqs))
        lo, hi = np.minimum(lens[i], lens[js]), np.maximum(lens[i], lens[js])
        for j in js[lo / hi >= threshold]:
            value = global_identity(aligner, seqs[i], seqs[j])
            if value >= threshold:
                out.append((i, int(j), value))
    return out


def compute_edges(
    sequences: Sequence[str],
    alphabet: TokenAlphabet,
    threshold: float,
    gap_open: float = 5,
    gap_extend: float = 1,
    n_jobs: int = 1,
) -> dict[tuple[int, int], float]:
    """Identity >= ``threshold`` for every pair; keys are (i, j) with i < j,
    positional indices into ``sequences`` (as QMAP's ``build_graph`` returns).
    Compute once at the lowest threshold of a sweep and filter the result."""
    sequences = list(sequences)
    args = (sequences, alphabet.chars, gap_open, gap_extend, threshold)
    n = len(sequences)
    chunks = [list(range(k, n, n_jobs * 8)) for k in range(n_jobs * 8)] if n_jobs > 1 else [list(range(n))]
    edges: dict[tuple[int, int], float] = {}
    if n_jobs <= 1:
        _init_worker(*args)
        for i, j, v in _rows(chunks[0]):
            edges[(i, j)] = v
        return edges
    with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=args) as pool:
        for part in pool.map(_rows, chunks):
            for i, j, v in part:
                edges[(i, j)] = v
    return edges


def contract_hard_links(
    n_nodes: int,
    edges: dict[tuple[int, int], float],
    hard_link_threshold: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Merge every pair with identity >= ``hard_link_threshold`` into one node.

    Community detection cuts weak edges, so it can separate two near-duplicate
    sequences. Contracting them first makes that impossible: they are one node
    and always land in the same cluster.

    Returns ``group`` (``group[i]`` = contracted node of original node ``i``,
    labelled 0..k-1) and the edge list of the contracted graph. **Every
    original edge is kept** (edges between two merged nodes stay as parallel
    edges; edges inside a merged node become self-loops), so each merged node
    keeps its original degree and the clustering sees how strongly two merged
    nodes were linked. Collapsing them to one simple edge was tried and made
    Leiden cut about twice as many above-threshold pairs.
    """
    hard = [pair for pair, value in edges.items() if value >= hard_link_threshold]
    group = np.array(ig.Graph(n=n_nodes, edges=hard, directed=False).connected_components().membership)
    return group, [(int(group[i]), int(group[j])) for i, j in edges]
