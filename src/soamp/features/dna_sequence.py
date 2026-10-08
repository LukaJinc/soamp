"""Pure helpers to pull an organism's 16S rRNA gene out of a RefSeq assembly:
FASTA records + GFF3 rRNA features -> sequence. No I/O beyond reading the two
already-downloaded files, no network, no model (the embedding itself lives in
soamp.features.dnabert_s)."""
from dataclasses import dataclass

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

# Full-length bacterial/archaeal 16S is ~1.5 kb; the bounds reject partial
# annotations and the rare mis-annotated feature.
MIN_16S_LENGTH = 1200
MAX_16S_LENGTH = 1700


class DnaSequenceError(ValueError):
    """Raised when no usable 16S rRNA gene can be extracted for an assembly."""


@dataclass(frozen=True)
class RrnaFeature:
    seqid: str
    start: int   # 1-based, inclusive (GFF3)
    end: int     # 1-based, inclusive
    strand: str
    product: str


def parse_fasta_records(path) -> list[tuple[str, str]]:
    """[(header_without_'>', uppercase_sequence), ...], one per FASTA record."""
    records: list[tuple[str, str]] = []
    header, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header, parts = line[1:], []
            else:
                parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def parse_gff_rrna_features(path) -> list[RrnaFeature]:
    """All `rRNA` features of a GFF3 file with their `product` attribute."""
    features = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 9 or columns[2] != "rRNA":
                continue
            attributes = dict(
                item.split("=", 1) for item in columns[8].split(";") if "=" in item
            )
            features.append(RrnaFeature(
                seqid=columns[0], start=int(columns[3]), end=int(columns[4]),
                strand=columns[6], product=attributes.get("product", ""),
            ))
    return features


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def extract_16s(records: list[tuple[str, str]], features: list[RrnaFeature]) -> str:
    """The organism's 16S rRNA gene sequence (5'->3' on the gene's own strand).

    Takes the 16S features whose length is in [MIN_16S_LENGTH, MAX_16S_LENGTH]
    and returns the longest, ties broken by genomic order: the copies of a
    genome's rRNA operons are near-identical, so any full-length copy is
    representative, and this rule is deterministic. Raises DnaSequenceError if
    the assembly has no full-length 16S annotation."""
    sequence_by_id = {header.split()[0]: seq for header, seq in records}
    candidates = []
    for feature in features:
        if "16S" not in feature.product:
            continue
        length = feature.end - feature.start + 1
        if not (MIN_16S_LENGTH <= length <= MAX_16S_LENGTH):
            continue
        contig = sequence_by_id.get(feature.seqid)
        if contig is None:
            continue
        gene = contig[feature.start - 1 : feature.end]
        candidates.append(reverse_complement(gene) if feature.strand == "-" else gene)
    if not candidates:
        raise DnaSequenceError(
            f"no full-length ({MIN_16S_LENGTH}-{MAX_16S_LENGTH} bp) 16S rRNA feature found"
        )
    return max(candidates, key=len)
