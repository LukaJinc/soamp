"""
Generates a peptide SMILES from a DBAASP sequence + unusualAminoAcids + intrachainBonds
+ terminus annotations, using p2smi as the underlying residue/bond SMILES engine.

Scope (deliberately bounded -- see bond_map.py for the full rationale):
  - Linear peptides: fully supported, for any sequence where every residue resolves
    (canonical L/D, or non-canonical via residue_map.lookup_p2smi).
  - Disulfide-bonded (SS) and head-to-tail cyclic (HT) peptides: supported, using the
    *exact* DBAASP-annotated bond positions (not p2smi's auto-inferred pairing).
  - Side-chain lactam bridges (SCSC/SCNT/SCCT) and any bond chemistry p2smi has no
    generator for (thioether/lanthionine, lactone/ester): NOT auto-generated here.
    Flagged unconvertible with a specific reason rather than guessed -- assigning which
    side-chain acts as amine-donor vs acid-acceptor is a real chemistry judgment call
    this pass does not attempt automatically.
  - N-terminal acetylation (ACT) and C-terminal amidation (AMD): applied by direct SMILES
    string edits, verified against known mass deltas (ACT +42, AMD -1 Da) matching
    QMAP's own modification_masses table. Any other terminus modification (lipidation,
    PEGylation, fluorescent/protecting groups, etc.) is NOT applied -- flagged unconvertible,
    since we have no verified SMILES fragment source for those groups.
"""
import p2smi.utilities.smilesgen as smilesgen
from soamp.curation.residue_map import lookup_p2smi
from soamp.curation.bond_map import classify_bond

LETTER2NAME = smilesgen.LETTER2NAME
SUPPORTED_NTERM = {None, "ACT"}
SUPPORTED_CTERM = {None, "AMD"}


def resolve_letters(sequence: str, unusual_aa: list):
    """
    Returns (letters: list[str] | None, unconvertible_reasons: list[str]).
    letters[i] is the p2smi internal Letter character for sequence position i+1.
    """
    letters = list(sequence)
    reasons = []
    unusual_positions = {pos: name for pos, name in unusual_aa}

    for i, ch in enumerate(letters):
        pos = i + 1
        if pos in unusual_positions:
            res = lookup_p2smi(unusual_positions[pos])
            if not res["matched"]:
                reasons.append(f"position {pos} ('{unusual_positions[pos]}'): {res['reason']}")
                continue
            letter = res["letter"]
            if res["is_d_form"]:
                letter = letter.lower()
                if letter not in LETTER2NAME:
                    reasons.append(f"position {pos} ('{unusual_positions[pos]}'): D-form letter "
                                    f"'{letter}' not recognised by p2smi -- no D-form entry available")
                    continue
            letters[i] = letter
        else:
            if ch not in LETTER2NAME:
                # Covers stray 'X' placeholders with no matching annotation, and any
                # other character p2smi's canonical (or D-canonical) table doesn't cover.
                reasons.append(f"position {pos} ('{ch}'): unresolved residue character, "
                                f"no unusualAminoAcids annotation covers this position")

    if reasons:
        return None, reasons
    return letters, []


def apply_termini(smi: str, nterminus: str, cterminus: str):
    if nterminus == "ACT":
        smi = "CC(=O)" + smi
    if cterminus == "AMD":
        smi = smi[:-1] + "N"
    return smi


def generate_smiles(sequence: str, unusual_aa: list, bonds: list, nterminus: str, cterminus: str):
    """
    Returns dict:
      {"smiles": str | None, "convertible": bool, "reason": str,
       "constraint_used": "linear"|"SS"|"HT"|None}
    """
    if nterminus not in SUPPORTED_NTERM:
        return {"smiles": None, "convertible": False, "constraint_used": None,
                 "reason": f"unsupported N-terminal modification '{nterminus}' "
                           f"(only free amine or ACT acetylation supported)"}
    if cterminus not in SUPPORTED_CTERM:
        return {"smiles": None, "convertible": False, "constraint_used": None,
                 "reason": f"unsupported C-terminal modification '{cterminus}' "
                           f"(only free acid or AMD amidation supported)"}

    letters, reasons = resolve_letters(sequence, unusual_aa)
    if letters is None:
        return {"smiles": None, "convertible": False, "constraint_used": None,
                 "reason": "; ".join(reasons)}

    if not bonds:
        smi = smilesgen.linear_peptide_smiles(letters)
        smi = apply_termini(smi, nterminus, cterminus)
        return {"smiles": smi, "convertible": True, "constraint_used": "linear", "reason": ""}

    if len(bonds) > 1:
        return {"smiles": None, "convertible": False, "constraint_used": None,
                 "reason": f"{len(bonds)} intrachain bonds present -- multi-bond (e.g. "
                           f"multiply-bridged / bicyclic) peptides are out of scope for "
                           f"automated generation in this pass"}

    bond = bonds[0]
    cls = classify_bond(bond)
    if not cls["convertible"] or cls["p2smi_constraint"] not in ("SS", "HT"):
        return {"smiles": None, "convertible": False, "constraint_used": None,
                 "reason": cls["reason"] if not cls["convertible"] else
                           f"bond constraint '{cls['p2smi_constraint']}' not auto-generated "
                           f"in this pass (requires side-chain donor/acceptor role assignment)"}

    if cls["p2smi_constraint"] == "HT":
        if nterminus == "ACT" or cterminus == "AMD":
            return {"smiles": None, "convertible": False, "constraint_used": None,
                     "reason": "head-to-tail cyclic bond present together with a terminus "
                               "modification (ACT/AMD) -- contradictory (termini are consumed "
                               "by the ring bond), flagged for manual review"}
        _, _, smi = smilesgen.constrained_peptide_smiles(letters, "HT")
        return {"smiles": smi, "convertible": True, "constraint_used": "HT", "reason": ""}

    if cls["p2smi_constraint"] == "SS":
        pos1, pos2 = bond["position1"], bond["position2"]
        # Sanity-check both bonded positions are actually disulfide-capable residues
        # before asking p2smi to build the bond -- DBAASP's own bond annotations
        # occasionally reference a position that isn't a Cys-like residue (a source
        # data inconsistency, observed directly during development), which would
        # otherwise surface as an opaque p2smi TypeError instead of a clear reason.
        for pos in (pos1, pos2):
            if pos < 1 or pos > len(letters):
                return {"smiles": None, "convertible": False, "constraint_used": None,
                         "reason": f"DSB bond position {pos} is outside the sequence "
                                   f"(length {len(letters)}) -- inconsistent DBAASP annotation"}
            letter = letters[pos - 1]
            name = LETTER2NAME.get(letter)
            props = smilesgen.all_aminos.get(name, {}) if name else {}
            if not props.get("disulphide"):
                return {"smiles": None, "convertible": False, "constraint_used": None,
                         "reason": f"DSB bond references position {pos} ('{sequence[pos-1]}' -> "
                                   f"{name}), which is not a disulfide-capable residue -- "
                                   f"inconsistent DBAASP bond annotation, flagged for manual review"}
        mask = "".join("C" if (i + 1) in (pos1, pos2) else "X" for i in range(len(letters)))
        pattern = "SS" + mask
        try:
            _, _, smi = smilesgen.constrained_peptide_smiles(letters, pattern)
        except Exception as e:
            return {"smiles": None, "convertible": False, "constraint_used": None,
                     "reason": f"p2smi failed to build SS-bonded SMILES at positions "
                               f"{pos1},{pos2}: {e}"}
        smi = apply_termini(smi, nterminus, cterminus)
        return {"smiles": smi, "convertible": True, "constraint_used": "SS", "reason": ""}

    return {"smiles": None, "convertible": False, "constraint_used": None,
             "reason": "unreachable"}
