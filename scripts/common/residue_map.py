"""
Maps DBAASP non-canonical residue codes (from `unusualAminoAcids[].modificationType.name`)
to p2smi's internal amino-acid database (p2smi.utilities.smilesgen.all_aminos), which is
keyed by a descriptive name and carries a "Code" (short abbreviation) and a "Letter"
(single internal unicode character used to build p2smi's pseudo-sequences).

Matching policy (conservative by design -- this feeds a published dataset):
  1. Exact match on Code (case-insensitive, "D-" stereo prefix stripped and recorded
     separately) against p2smi's Code field.
  2. A small hand-verified synonym table for cases where DBAASP and p2smi use different
     abbreviations for the unambiguously same residue (verified by cross-checking the
     chemical name / molecular formula, not guessed).
  3. Anything not matched by (1) or (2) is left UNMAPPED. We deliberately do not do
     fuzzy/best-guess name matching for the general case: a wrong structural match here
     would corrupt SMILES and molecular weight for a published dataset, which is worse
     than excluding the entry and flagging it.

Known DBAASP terminal-modification codes are handled separately (see TERMINAL_* below) --
none of these are peptide "residues" and none currently have a reliable programmatic SMILES
fragment source (lipidation chain length, PEG size, fluorophores, protecting groups).
"""
import p2smi.utilities.smilesgen as smilesgen

ALL_AMINOS = smilesgen.all_aminos
LETTER2NAME = smilesgen.LETTER2NAME

# Build Code -> (name, props) index from p2smi's database.
_P2SMI_BY_CODE = {}
for _name, _props in ALL_AMINOS.items():
    code = _props.get("Code")
    if code:
        _P2SMI_BY_CODE.setdefault(code.upper(), []).append((_name, _props))

# Hand-verified synonym table: DBAASP code (upper, stereo-stripped) -> p2smi Code.
# Each entry below was checked interactively against p2smi's Formula/SMILES for the target
# Code to confirm it is the same molecule DBAASP's modification name refers to (matching
# chemical formula), not a name-similarity guess. ORN, DAB and NLE need no entry here --
# they already match p2smi's Code field directly (see _P2SMI_BY_CODE).
#
# Verification notes (formula cross-check performed against p2smi.utilities.smilesgen.all_aminos):
#   ABU   -> ABA   : both "2-Aminobutyric-acid", C4H9NO2
#   CHA   -> ALC   : Cha = 3-cyclohexyl-alanine = p2smi "3-cyclohexyl-alanine", C9H17NO2
#   CIT   -> CIR   : both "Citrulline", C6H13N3O3
#   HARG  -> HRG   : both "homoarginine", C7H18N4O2
#   PHG   -> PG    : Phg = Phenylglycine = p2smi "Phenylglycine", C8H9NO2
#   1-NAL -> ALN   : both "1-Naphthyl-alanine", C13H13NO2
#   MET(O)-> SME   : Met(O) = methionine sulfoxide = p2smi "Methionine-sulfoxide", C5H11NO3S
#   NVAL  -> NVA   : both "Norvaline", C5H11NO2
DBAASP_TO_P2SMI_SYNONYMS = {
    "ABU": "ABA",
    "CHA": "ALC",
    "CIT": "CIR",
    "HARG": "HRG",
    "PHG": "PG",
    "1-NAL": "ALN",
    "MET(O)": "SME",
    "NVAL": "NVA",
}
# NOTE: this table is a best-effort curated subset built from the residue codes observed
# during the DBAASP crawl for this project, not an exhaustive DBAASP<->p2smi mapping.
# Any DBAASP code not covered here (and not an exact Code match) falls through to
# `lookup_p2smi`'s "unmapped" path and is logged as unconvertible with the specific code,
# per the project's no-silent-drop / no-nearest-analog-substitution requirement.


def normalize_dbaasp_code(raw_code: str):
    """
    Returns (base_code_upper, is_d_form: bool).
    DBAASP marks D-stereochemistry with a "D-" prefix (e.g. "D-ORN", "D-Allo-ILE").
    """
    code = raw_code.strip()
    is_d = False
    if code.upper().startswith("D-"):
        is_d = True
        code = code[2:]
    return code.upper(), is_d


def lookup_p2smi(raw_dbaasp_code: str):
    """
    Attempt to resolve a DBAASP unusualAminoAcids modificationType.name to a p2smi
    residue entry.

    Returns a dict:
      {
        "matched": bool,
        "base_code": str,
        "is_d_form": bool,
        "p2smi_name": str | None,
        "letter": str | None,       # the character to substitute for L-form; caller
                                     # lowercases it for D-form (p2smi convention)
        "reason": str,               # explanation, used when matched=False
      }
    """
    base_code, is_d = normalize_dbaasp_code(raw_dbaasp_code)

    candidates = _P2SMI_BY_CODE.get(base_code)
    p2smi_code_to_use = base_code

    if not candidates:
        synonym = DBAASP_TO_P2SMI_SYNONYMS.get(base_code)
        if synonym:
            p2smi_code_to_use = synonym
            candidates = _P2SMI_BY_CODE.get(synonym)

    if not candidates:
        return {
            "matched": False, "base_code": base_code, "is_d_form": is_d,
            "p2smi_name": None, "letter": None,
            "reason": f"unknown residue code '{raw_dbaasp_code}': no match in p2smi database "
                      f"(checked Code='{base_code}'" +
                      (f" and synonym Code='{p2smi_code_to_use}'" if p2smi_code_to_use != base_code else "") + ")",
        }

    if len(candidates) > 1:
        return {
            "matched": False, "base_code": base_code, "is_d_form": is_d,
            "p2smi_name": None, "letter": None,
            "reason": f"ambiguous residue code '{raw_dbaasp_code}': {len(candidates)} p2smi "
                      f"entries share Code='{p2smi_code_to_use}' ({[c[0] for c in candidates]}) "
                      f"-- flagged for manual review rather than guessing",
        }

    name, props = candidates[0]
    return {
        "matched": True, "base_code": base_code, "is_d_form": is_d,
        "p2smi_name": name, "letter": props["Letter"],
        "reason": "",
    }
