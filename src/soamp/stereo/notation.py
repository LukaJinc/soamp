"""Stereochemistry as written in DBAASP's sequence notation.

Two independent notation signals exist and are kept apart (never merged):
  * case of the character in ``sequence`` -- lowercase = D (canonical residues
    and also unusual ``x``);
  * the ``D-`` prefix of ``unusualAminoAcids[].modificationType.name`` (for the
    unusual residues only).
Achiral residues (Gly, Aib, N-substituted glycines) are decided from a
reference SMILES of the residue, not from the name.
"""
from collections.abc import Mapping, Sequence

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def case_stereo(char: str) -> str:
    """'D' for lowercase, 'L' for uppercase (Gly handled by the caller)."""
    return "D" if char.islower() else "L"


def name_stereo(name: str | None) -> str | None:
    """'D' if DBAASP's residue name starts with ``D-`` (case-insensitive),
    else 'L'. ``None`` when there is no name."""
    if not name:
        return None
    return "D" if name.strip().upper().startswith("D-") else "L"


def unusual_by_position(unusual: Sequence[Mapping] | None) -> tuple[dict[int, dict], list[int]]:
    """position (1-based) -> {name, template_smiles}; second value lists
    positions that carry two *different* names (ambiguous, reported)."""
    out: dict[int, dict] = {}
    conflicts: list[int] = []
    for entry in unusual or []:
        mod = entry.get("modificationType") or {}
        name, pos = mod.get("name"), entry.get("position")
        if not name or pos is None:
            continue
        pos = int(pos)
        if pos in out and out[pos]["name"] != name:
            conflicts.append(pos)
            continue
        out[pos] = {"name": name, "template_smiles": mod.get("smiles")}
    return out, conflicts


def parse_notation(sequence: str | None, unusual: Sequence[Mapping] | None) -> list[dict]:
    """One dict per sequence position with the notation-side classification.

    Keys: position, char, residue_class (standard | nonstandard | unrecognized),
    token, case_stereo, name_stereo, notation_stereo, notation_reason.
    ``notation_stereo`` is the case signal for standard residues and the name
    signal for unusual ones; when case and name disagree it is ``undetermined``
    (both raw signals stay in their own columns).
    """
    if not sequence:
        return []
    by_pos, conflicts = unusual_by_position(unusual)
    rows = []
    for pos, ch in enumerate(sequence, start=1):
        row = {"position": pos, "char": ch, "case_stereo": None, "name_stereo": None,
               "template_smiles": None, "notation_reason": ""}
        up = ch.upper()
        if up in CANONICAL and pos not in by_pos:
            row.update(residue_class="standard", token=up,
                       case_stereo=case_stereo(ch),
                       notation_stereo="achiral" if up == "G" else case_stereo(ch))
        elif up == "X" and pos in by_pos and pos not in conflicts:
            info = by_pos[pos]
            row.update(residue_class="nonstandard", token=info["name"],
                       template_smiles=info["template_smiles"],
                       case_stereo=case_stereo(ch), name_stereo=name_stereo(info["name"]))
            row["notation_stereo"] = row["name_stereo"]
            if row["case_stereo"] != row["name_stereo"]:
                row["notation_stereo"] = "undetermined"
                row["notation_reason"] = (f"case says {row['case_stereo']} but name "
                                          f"'{info['name']}' says {row['name_stereo']}")
        else:
            reason = (f"two different residue names at position {pos}" if pos in conflicts else
                      f"'{ch}' is not a canonical letter and has no unusualAminoAcids entry"
                      if up == "X" or up not in CANONICAL else
                      f"canonical letter '{ch}' overridden by an unusualAminoAcids entry")
            row.update(residue_class="unrecognized", token=ch, notation_stereo="undetermined",
                       notation_reason=reason)
        rows.append(row)
    return rows
