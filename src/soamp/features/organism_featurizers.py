"""Pluggable organism featurization strategies, selected by a plain method
name via a Protocol + concrete-classes + one build_*() dispatch function
pattern (a dedicated error type raised on an unknown method).

output_kind tells src/soamp/model/factory.py::build_model whether the
model should own a learned nn.Embedding ("index") or an nn.Linear
projection ("vector") for the organism input -- see
src/soamp/model/baseline_mlp.py::OrganismEncoder. Two methods ship today:
`vocab_embedding` (output_kind="index") and `kmer_composition`
(output_kind="vector", genome k-mer composition, modeled on LLAMP) -- both
implement the same protocol with no model-side changes needed to add
either.
"""
import csv
from pathlib import Path
from typing import Literal, Protocol

from soamp.features.kmer import compute_kmer_composition, parse_fasta_sequences
from soamp.features.organism import build_vocab, encode

REPO_ROOT = Path(__file__).resolve().parents[3]


class OrganismFeaturizerError(ValueError):
    """Raised when an unknown organism featurization method is requested,
    or when a fit-group organism has no resolvable representation."""


class OrganismFeaturizer(Protocol):
    output_kind: Literal["index", "vector"]

    def fit(self, fit_organisms: list[str]) -> None: ...

    @property
    def vocab_size(self) -> int | None:
        """Meaningful iff output_kind == 'index'."""
        ...

    @property
    def feature_dim(self) -> int | None:
        """Meaningful iff output_kind == 'vector'."""
        ...

    def encode(self, organism: str) -> "int | list[float]": ...

    def to_artifact_dict(self) -> dict:
        """Self-describing serialization (always includes a "method" key)
        consumed by pipeline/features/02_build_organism_vocab.py to write
        data/organism_vocab.json generically, and by
        organism_featurizer_from_artifact() to reconstruct a featurizer
        from that file without recomputing/re-fetching anything."""
        ...


class VocabEmbeddingOrganismFeaturizer:
    """Thin adapter over soamp.features.organism's existing pure functions.
    `vocab` is settable directly (bypassing fit()) so artifact-mode dataset
    construction can hydrate it from a pre-built organism_vocab.json without
    re-deriving it from row data."""

    output_kind: Literal["index"] = "index"

    def __init__(self, unknown_index: int = 0, vocab: dict[str, int] | None = None) -> None:
        self.unknown_index = unknown_index
        self.vocab = vocab

    def fit(self, fit_organisms: list[str]) -> None:
        self.vocab = build_vocab(fit_organisms, unknown_index=self.unknown_index)

    @property
    def vocab_size(self) -> int | None:
        return None if self.vocab is None else len(self.vocab) + 1

    @property
    def feature_dim(self) -> int | None:
        return None

    def encode(self, organism: str) -> int:
        return encode(self.vocab, organism, self.unknown_index)

    def to_artifact_dict(self) -> dict:
        return {
            "method": "vocab_embedding",
            "unknown_index": self.unknown_index,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
        }


class KmerOrganismFeaturizer:
    """Genome k-mer composition organism representation, modeled on LLAMP
    (GIST-CSBL/LLAMP): mono/di/tri/tetra-nucleotide composition of a
    representative RefSeq genome assembly per organism, each k's count
    vector L2-normalized separately then concatenated -- see
    soamp.features.kmer.compute_kmer_composition for the exact computation
    (standard sliding-window counting, not LLAMP's own str.count-based
    counting). Unlike VocabEmbeddingOrganismFeaturizer, a genome vector is
    a fixed external constant per species -- it doesn't depend on which
    rows are in the fit split, so fit() computes/validates coverage rather
    than restricting what encode() can later resolve (the point of a
    genome-based representation is generalizing to organisms never seen in
    fit, unlike a learned embedding's uninformative shared OOV index).

    Species-exact match -> genus fallback (first word) -> all-zero vector,
    mirroring soamp.common.thresholds.lookup_threshold's exact precedence.
    """

    output_kind: Literal["vector"] = "vector"

    def __init__(
        self,
        k_values: tuple[int, ...] = (1, 2, 3, 4),
        accessions_csv_path: str = "config/organism_genomes/organism_genome_accessions.csv",
        genome_cache_dir: str = ".cache/refseq_genomes",
        species_features: dict[str, list[float]] | None = None,
        genus_features: dict[str, list[float]] | None = None,
    ) -> None:
        self.k_values = tuple(k_values)
        self.accessions_csv_path = self._resolve_path(accessions_csv_path)
        self.genome_cache_dir = self._resolve_path(genome_cache_dir)
        self._species_features = species_features
        self._genus_features = genus_features

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def _is_loaded(self) -> bool:
        return self._species_features is not None and self._genus_features is not None

    def _ensure_loaded(self) -> None:
        if self._is_loaded:
            return
        species_features: dict[str, list[float]] = {}
        genus_features: dict[str, list[float]] = {}
        with open(self.accessions_csv_path, newline="") as f:
            for row in csv.DictReader(f):
                accession = (row.get("refseq_assembly_accession") or "").strip()
                if not accession:
                    continue
                match_key = row["match_key"].strip()
                level = row["level"].strip()
                fasta_path = self.genome_cache_dir / f"{accession}.fasta"
                sequences = parse_fasta_sequences(fasta_path)
                composition = compute_kmer_composition(sequences, self.k_values)
                vector = list(composition.values())
                if level == "species":
                    species_features[match_key] = vector
                elif level == "genus":
                    genus_features[match_key] = vector
        self._species_features = species_features
        self._genus_features = genus_features

    def fit(self, fit_organisms: list[str]) -> None:
        self._ensure_loaded()
        missing = sorted({o for o in fit_organisms if self._resolve(o) is None})
        if missing:
            raise OrganismFeaturizerError(
                f"no genome k-mer features available for organisms: {missing}"
            )

    @property
    def vocab_size(self) -> int | None:
        return None

    @property
    def feature_dim(self) -> int | None:
        return sum(4**k for k in self.k_values)

    def _resolve(self, organism: str) -> list[float] | None:
        organism = organism.strip()
        if organism in self._species_features:
            return self._species_features[organism]
        genus = organism.split(" ", 1)[0] if organism else ""
        if genus in self._genus_features:
            return self._genus_features[genus]
        return None

    def encode(self, organism: str) -> list[float]:
        self._ensure_loaded()
        vector = self._resolve(organism)
        return vector if vector is not None else [0.0] * self.feature_dim

    def to_artifact_dict(self) -> dict:
        self._ensure_loaded()
        return {
            "method": "kmer_composition",
            "k_values": list(self.k_values),
            "feature_dim": self.feature_dim,
            "species_features": self._species_features,
            "genus_features": self._genus_features,
        }


def build_organism_featurizer(method: str, **method_kwargs) -> OrganismFeaturizer:
    if method == "vocab_embedding":
        return VocabEmbeddingOrganismFeaturizer(**method_kwargs)
    if method == "kmer_composition":
        return KmerOrganismFeaturizer(**method_kwargs)
    raise OrganismFeaturizerError(f"unknown organism featurization method: {method!r}")


def organism_featurizer_from_artifact(data: dict) -> OrganismFeaturizer:
    """Reconstructs a fitted OrganismFeaturizer directly from a loaded
    organism_vocab.json-shaped dict (self-describing via its "method" key),
    without re-fetching/recomputing anything -- used by
    src/soamp/data/factory.py::_build_dataset_from_artifacts."""
    method = data["method"]
    if method == "vocab_embedding":
        return VocabEmbeddingOrganismFeaturizer(
            unknown_index=data["unknown_index"], vocab=data["vocab"],
        )
    if method == "kmer_composition":
        return KmerOrganismFeaturizer(
            k_values=tuple(data["k_values"]),
            species_features=data["species_features"],
            genus_features=data["genus_features"],
        )
    raise OrganismFeaturizerError(f"unknown organism featurization method in artifact: {method!r}")
