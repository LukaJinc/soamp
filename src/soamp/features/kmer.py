"""Pure genome k-mer composition computation -- no I/O beyond parsing an
already-read FASTA string/file, no network. Mirrors soamp.features.peptide's
pure-function philosophy: given sequence data, compute a fixed-length named
feature vector, nothing more.

Modeled on LLAMP (GIST-CSBL/LLAMP)'s genome feature engineering -- verified
against their actual code (utils/utils.py::count_mer / get_features), not
just the paper: mono/di/tri/tetra-nucleotide (k=1,2,3,4) composition,
4+16+64+256=340 features, each k's count vector L2-normalized separately
before concatenation. Deliberately NOT replicating LLAMP's own k-mer
counting implementation, which uses Python's `str.count(kmer)` -- a
non-overlapping substring count that undercounts overlapping repeats (e.g.
"AAAA".count("AA") == 2, not the 3 overlapping occurrences a sliding window
finds). This module uses standard sliding-window counting instead.
"""
from itertools import product

DNA_ALPHABET = "ACGT"


def parse_fasta_sequences(path) -> list[str]:
    """One string per FASTA record (header lines starting with '>' stripped,
    sequence lines concatenated, uppercased). No biopython dependency --
    this format is simple enough not to need one. A genome assembly with
    multiple contigs/chromosomes/plasmids yields one string per record;
    compute_kmer_composition sums counts across all of them."""
    sequences: list[str] = []
    current: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line.upper())
    if current:
        sequences.append("".join(current))
    return sequences


def _l2_normalize(values: list[float]) -> list[float]:
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0.0:
        return values
    return [v / norm for v in values]


def compute_kmer_composition(
    sequences: list[str], k_values: tuple[int, ...] = (1, 2, 3, 4)
) -> dict[str, float]:
    """Standard sliding-window k-mer counting (every overlapping window,
    across all sequences/contigs combined) for each k in k_values, skipping
    any window containing a base outside A/C/G/T. Each k's count vector is
    L2-normalized separately (matching LLAMP's per-block normalization),
    then all blocks are concatenated into one named dict, e.g.
    {"1mer_A": ..., "2mer_AC": ..., ..., "4mer_TTTT": ...} -- 340 names for
    k_values=(1,2,3,4). All-zero counts for a block (e.g. no sequence data)
    stay all-zero after normalization rather than dividing by zero.
    """
    result: dict[str, float] = {}
    for k in k_values:
        kmers = ["".join(t) for t in product(DNA_ALPHABET, repeat=k)]
        counts = {kmer: 0 for kmer in kmers}
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                window = seq[i : i + k]
                if window in counts:
                    counts[window] += 1
        normalized = _l2_normalize([float(counts[kmer]) for kmer in kmers])
        for kmer, value in zip(kmers, normalized):
            result[f"{k}mer_{kmer}"] = value
    return result
