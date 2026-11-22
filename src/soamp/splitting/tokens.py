"""Tokenization of linear DBAASP peptides for sequence alignment.

DBAASP writes every unusual amino acid as ``X`` in ``sequence`` and says
which one in ``unusualAminoAcids[]`` (``position`` is 1-based,
``modificationType.name`` is the residue). QMAP's compiled aligner cannot
distinguish different ``X`` residues (see ``alignment.py``), so each peptide
is turned into a list of *tokens*: the canonical one-letter code, or
``"mod:<name>"`` for an unusual residue.

The modification name is used exactly as DBAASP spells it (free text such as
``3,4-OH-Arg`` or ``Npm(p-F)``): no case-folding, no parsing, no merging of
spellings, and SMILES is ignored.
"""
from collections.abc import Iterable, Mapping, Sequence

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
MOD_PREFIX = "mod:"


class TokenizationError(ValueError):
    """Raised by strict tokenization when an ``X`` cannot be resolved."""


def tokenize_linear(sequence: str, unusual: Sequence[Mapping] | None) -> list[str] | None:
    """Tokens of one linear peptide, or ``None`` if it cannot be resolved.

    Canonical letters are upper-cased (matching QMAP, which treats ``a`` and
    ``A`` alike). ``X`` at 1-based position ``p`` becomes
    ``"mod:" + modificationType.name`` of the entry with ``position == p``.
    ``None`` is returned when an ``X`` has no entry, when entries for one
    position disagree, or when the sequence holds any other character.
    """
    by_position: dict[int, str] = {}
    for entry in unusual or []:
        name = (entry.get("modificationType") or {}).get("name")
        position = entry.get("position")
        if not name or position is None:
            continue
        if by_position.setdefault(int(position), name) != name:
            return None  # two different residues claimed for one position
    tokens: list[str] = []
    for index, char in enumerate(sequence.upper(), start=1):
        if char in CANONICAL:
            tokens.append(char)
        elif char == "X" and index in by_position:
            tokens.append(MOD_PREFIX + by_position[index])
        else:
            return None
    return tokens


def tokenize_linear_strict(sequence: str, unusual: Sequence[Mapping] | None) -> list[str]:
    """Like :func:`tokenize_linear` but raises :class:`TokenizationError`."""
    tokens = tokenize_linear(sequence, unusual)
    if tokens is None:
        raise TokenizationError(f"cannot tokenize sequence {sequence!r}")
    return tokens


def build_modification_table(token_lists: Iterable[Sequence[str]]) -> dict[str, int]:
    """Occurrence count per distinct ``mod:<name>`` token, most frequent first
    (ties broken by name so the table is deterministic). Printed in the
    notebook for review."""
    counts: dict[str, int] = {}
    for tokens in token_lists:
        for token in tokens:
            if token.startswith(MOD_PREFIX):
                counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
