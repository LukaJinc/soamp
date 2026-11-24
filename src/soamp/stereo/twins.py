"""Stereo-only twins: peptides identical once every D-residue is mapped to its L-form."""
from collections import defaultdict
from itertools import combinations

D_PREFIX = "D-"


def to_l_token(token: str, char: str) -> str:
    """Notation token with the D-form mapped to L: canonical -> upper-case letter,
    unusual -> name without its ``D-`` prefix (``D-ORN`` -> ``ORN``)."""
    if len(char) == 1 and char.upper() in "ACDEFGHIKLMNPQRSTVWY" and token == char.upper():
        return token
    name = token.strip()
    return name[len(D_PREFIX):] if name.upper().startswith(D_PREFIX) else name


def twin_pairs(records: list[dict]) -> list[dict]:
    """records: {peptide_id, mapped (tuple), stereo (tuple), annotation (hashable)}.
    Pairs sharing ``mapped`` but with different ``stereo``; ``same_annotation`` says
    whether termini/bonds also agree."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["mapped"]].append(r)
    pairs = []
    for members in groups.values():
        for a, b in combinations(members, 2):
            if a["stereo"] != b["stereo"]:
                pairs.append({"peptide_a": a["peptide_id"], "peptide_b": b["peptide_id"],
                              "same_annotation": a["annotation"] == b["annotation"]})
    return pairs
