"""Per-residue / per-peptide stereo audit: notation vs SMILES, kept side by side."""
from collections import Counter
from collections.abc import Mapping

from rdkit import Chem

from soamp.stereo.match import (
    alpha_sites, build_template, chain_order, corn_stereo, match_template,
    standard_templates, template_stereo,
)
from soamp.stereo.notation import parse_notation

CHIRAL = ("L", "D")


def _template_for(row: dict, cache: dict):
    """Reference template of one residue: the standard L-residue, or DBAASP's own SMILES."""
    if row["residue_class"] == "standard":
        return standard_templates().get(row["token"])
    if row["residue_class"] == "nonstandard":
        key = row["token"]
        if key not in cache:
            cache[key] = build_template(key, row.get("template_smiles"))
        return cache[key]
    return None


def audit_peptide(peptide_id, sequence: str | None, smiles: str | None,
                  unusual: list | None, template_cache: dict) -> tuple[list[dict], dict]:
    """Residue rows and a peptide-level info dict (mapping status, topology, SMILES counts)."""
    notation = parse_notation(sequence, unusual)
    info = {"peptide_id": peptide_id, "length": len(notation), "n_alpha_smiles": None,
            "topology": None, "mapping_status": "", "smiles_counts": None}
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    sites, order = [], None
    if not notation:
        info["mapping_status"] = "no_sequence"
    elif not smiles:
        info["mapping_status"] = "no_smiles"
    elif mol is None:
        info["mapping_status"] = "smiles_parse_failed"
    else:
        sites = alpha_sites(mol)
        order, info["topology"] = chain_order(mol, sites)
        info["n_alpha_smiles"] = len(sites)
        if order is not None and len(order) == len(notation):
            info["mapping_status"] = "positional"
        elif len(sites) != len(notation):
            info["mapping_status"] = f"count_mismatch(smiles={len(sites)},sequence={len(notation)})"
        else:
            info["mapping_status"] = f"unordered_{info['topology']}"
        # peptide-level stereo counts from the SMILES, usable even without positions
        info["smiles_counts"] = Counter(corn_stereo(mol, s)[0] for s in sites)

    rows = []
    for i, nrow in enumerate(notation):
        row = {"peptide_id": peptide_id, **{k: v for k, v in nrow.items() if k != "template_smiles"},
               "smiles_stereo_corn": None, "smiles_stereo_template": None,
               "template_relation": None, "smiles_reason": ""}
        if info["mapping_status"] == "positional":
            site = order[i]
            row["smiles_stereo_corn"], why = corn_stereo(mol, site)
            row["smiles_reason"] = why
            tpl = _template_for(nrow, template_cache)
            if tpl is None:
                row["smiles_stereo_template"] = "undetermined"
                row["smiles_reason"] = (row["smiles_reason"] + "; no reference template").strip("; ")
            else:
                rel = match_template(mol, site, tpl)
                row["template_relation"] = rel
                row["smiles_stereo_template"] = template_stereo(tpl, rel) if rel else "undetermined"
                if rel is None:
                    row["smiles_reason"] = (row["smiles_reason"] + "; residue does not match its template").strip("; ")
        else:
            row["smiles_reason"] = info["mapping_status"]
        rows.append(row)
    return rows, info


def peptide_label(stereos: list[str | None], ignore_undetermined: bool = False) -> str:
    """all_L | all_D | mixed | no_chiral_residues | undetermined."""
    known = {s for s in stereos if s in CHIRAL}
    unknown = any(s not in CHIRAL + ("achiral",) for s in stereos)
    if unknown and not ignore_undetermined:
        return "undetermined"
    if not known:
        return "no_chiral_residues" if not unknown else "undetermined"
    return "mixed" if len(known) == 2 else f"all_{next(iter(known))}"


def residue_agreement(row: Mapping) -> str:
    """notation vs SMILES (backbone-role method) for one residue.

    ``notation_default_L_but_achiral``: an unusual residue whose name has no ``D-``
    prefix counts as L in the notation only by default; if its SMILES alpha carbon is
    achiral (e.g. N-substituted glycines such as NLys) the notation had nothing to say
    -- reported separately, not as a real L/D conflict.
    """
    a, b = row["notation_stereo"], row["smiles_stereo_corn"]
    if b is None or b != b:   # None or NaN (after a parquet round-trip)
        return "smiles_unavailable"
    if a == "undetermined" or b == "undetermined":
        return "undetermined"
    if a == b:
        return "agree"
    if row["residue_class"] == "nonstandard" and a == "L" and b == "achiral":
        return "notation_default_L_but_achiral"
    return "disagree"
