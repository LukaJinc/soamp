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
from typing import Literal

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
MOD_PREFIX = "mod:"

DResidues = Literal["separate", "map_to_L"]

# ``map_to_L`` strips a leading ``D-`` from a residue name. These names hold the
# stereo marker elsewhere (``4-NH2-D-Pro``) or collide in letter case once the
# prefix is gone (``D-TIC`` -> ``TIC``, but the L-form is spelled ``Tic``), so
# they are mapped explicitly. Found by auditing every name in the dev corpus.
DEFAULT_NAME_EXCEPTIONS: dict[str, str] = {
    "4-NH2-D-Pro": "4-NH2-Pro",
    "cis-4-D-Hyp": "cis-4-Hyp",
    "C10-N6-D-Lys": "C10-N6-Lys",
    "β-Thi-D-Ala": "β-Thi-Ala",
    "D-TIC": "Tic",
}


class TokenizationError(ValueError):
    """Raised by strict tokenization when an ``X`` cannot be resolved."""


def to_l_name(name: str, name_exceptions: Mapping[str, str] | None = None) -> str:
    """L-form of a DBAASP residue name: exact-match exception table first, else a
    leading ``D-`` (case-insensitive) is dropped. Names are never case-folded
    otherwise, and R/S, cis/trans and ``Allo-`` descriptors are left alone (they
    are not D/L labels)."""
    exceptions = DEFAULT_NAME_EXCEPTIONS if name_exceptions is None else name_exceptions
    if name in exceptions:
        return exceptions[name]
    return name[2:] if name[:2].upper() == "D-" else name


def tokenize_linear(
    sequence: str,
    unusual: Sequence[Mapping] | None,
    d_residues: DResidues = "separate",
    name_exceptions: Mapping[str, str] | None = None,
) -> list[str] | None:
    """Tokens of one linear peptide, or ``None`` if it cannot be resolved.

    Canonical letters are upper-cased (matching QMAP, which treats ``a`` and
    ``A`` alike). ``X`` at 1-based position ``p`` becomes
    ``"mod:" + modificationType.name`` of the entry with ``position == p``.
    ``None`` is returned when an ``X`` has no entry, when entries for one
    position disagree, or when the sequence holds any other character.

    ``d_residues="separate"`` (default) keeps ``D-Orn`` and ``Orn`` as two
    different tokens; ``"map_to_L"`` converts every D-form residue name to its
    L-form first (see :func:`to_l_name`), so a peptide and its D-substituted
    variant align as the same sequence. Canonical D-residues are always mapped
    (upper-casing). Entries are compared **after** mapping, so ``D-Orn`` and
    ``Orn`` at one position still count as one residue.
    """
    if d_residues not in ("separate", "map_to_L"):
        raise ValueError(f"d_residues must be 'separate' or 'map_to_L', got {d_residues!r}")
    by_position: dict[int, str] = {}
    for entry in unusual or []:
        name = (entry.get("modificationType") or {}).get("name")
        position = entry.get("position")
        if not name or position is None:
            continue
        if d_residues == "map_to_L":
            name = to_l_name(name, name_exceptions)
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


def tokenize_linear_strict(
    sequence: str,
    unusual: Sequence[Mapping] | None,
    d_residues: DResidues = "separate",
    name_exceptions: Mapping[str, str] | None = None,
) -> list[str]:
    """Like :func:`tokenize_linear` but raises :class:`TokenizationError`."""
    tokens = tokenize_linear(sequence, unusual, d_residues, name_exceptions)
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
