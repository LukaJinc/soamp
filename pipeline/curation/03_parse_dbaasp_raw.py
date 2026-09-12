"""
Step 2 (part b): flatten the raw DBAASP crawl (.cache/dbaasp_raw.jsonl, one JSON
object per peptide, native API shape) into data/dbaasp_raw_full.csv (one row per
DBAASP peptide entry -- "entries pulled"), and log the audit counts the task asks
for: total entries, entries with native SMILES, non-canonical residue codes
detected (with counts), and intrachain bond types other than disulfide/amide
(with counts).

No filtering happens here -- every peptide entry successfully fetched is retained
and flagged, per the task's "keep everything, flag it instead" instruction.
"""
import argparse
import os
import json
import csv
from collections import Counter

from dotenv import load_dotenv

from soamp.curation.config import CurationConfig
from soamp.curation.parse_dbaasp import DBAASPPeptide
from soamp.utils.config import load_config


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/curation/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, CurationConfig)
RAW_JSONL = CFG.paths.cache_dir / "dbaasp_raw.jsonl"
OUT_CSV = CFG.paths.data_dir / "dbaasp_raw_full.csv"
LOG_PATH = CFG.paths.reports_dir / "step2_dbaasp_raw_pull_log.txt"

FIELDNAMES = [
    "peptide_id", "dbaasp_id", "sequence", "sequence_length",
    "nterminus", "cterminus", "synthesis_type", "complexity",
    "target_groups", "target_objects",
    "n_unusual_aa", "unusual_aa_detail", "has_noncanonical", "has_d_residue",
    "n_intrachain_bonds", "bond_types", "bond_cycletypes",
    "has_native_smiles", "n_native_smiles", "smiles_native",
    "n_target_activities", "n_mic_bacterial_activities_raw", "species_list",
]


def main():
    rows = []
    n_total = 0
    n_with_smiles = 0
    unusual_aa_counter = Counter()
    non_std_bond_counter = Counter()  # bond types other than DSB/AMD
    all_bond_type_counter = Counter()
    complexity_counter = Counter()

    with open(RAW_JSONL) as f:
        for line in f:
            n_total += 1
            raw = json.loads(line)
            p = DBAASPPeptide(raw)

            if p.has_native_smiles:
                n_with_smiles += 1

            for _, mod_name in p.unusual_amino_acids:
                unusual_aa_counter[mod_name] += 1

            complexity_counter[p.complexity] += 1

            for bt in p.bond_types:
                all_bond_type_counter[bt] += 1
                if bt not in ("DSB", "AMD"):
                    non_std_bond_counter[bt] += 1

            species = sorted({(t.get("targetSpecies") or {}).get("name")
                               for t in p.target_activities
                               if (t.get("targetSpecies") or {}).get("name")})
            n_mic_bacterial_raw = sum(
                1 for t in p.target_activities
                if (t.get("activityMeasureGroup") or {}).get("name") == "MIC"
            )

            rows.append({
                "peptide_id": p.id,
                "dbaasp_id": p.dbaasp_id,
                "sequence": p.sequence,
                "sequence_length": p.sequence_length,
                "nterminus": p.nterminus,
                "cterminus": p.cterminus,
                "synthesis_type": p.synthesis_type,
                "complexity": p.complexity,
                "target_groups": "|".join(p.target_groups),
                "target_objects": "|".join(p.target_objects),
                "n_unusual_aa": len(p.unusual_amino_acids),
                "unusual_aa_detail": "|".join(f"{pos}:{name}" for pos, name in p.unusual_amino_acids),
                "has_noncanonical": p.has_noncanonical,
                "has_d_residue": p.has_d_residue,
                "n_intrachain_bonds": len(p.intrachain_bonds),
                "bond_types": "|".join(p.bond_types),
                "bond_cycletypes": "|".join(sorted({
                    (b.get("cycleType") or {}).get("name") for b in p.intrachain_bonds
                    if (b.get("cycleType") or {}).get("name")
                })),
                "has_native_smiles": p.has_native_smiles,
                "n_native_smiles": len(p.native_smiles_list),
                "smiles_native": "|".join(p.native_smiles_list),
                "n_target_activities": len(p.target_activities),
                "n_mic_bacterial_activities_raw": n_mic_bacterial_raw,
                "species_list": "|".join(species),
            })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_noncanonical = sum(1 for r in rows if r["has_noncanonical"])
    n_with_nonstd_bond = sum(1 for r in rows if any(bt not in ("DSB", "AMD") for bt in r["bond_types"].split("|") if bt))

    log_lines = []
    log_lines.append("=== Step 2: DBAASP raw pull (direct REST API) ===")
    log_lines.append(f"Endpoint: GET https://dbaasp.org/peptides/{{id}}, Accept: application/json")
    log_lines.append(f"Total entries successfully pulled: {n_total}")
    log_lines.append(f"Entries with native SMILES field populated: {n_with_smiles} ({100*n_with_smiles/n_total:.1f}%)")
    log_lines.append(f"Entries flagged has_noncanonical: {n_noncanonical} ({100*n_noncanonical/n_total:.1f}%)")
    log_lines.append("")
    log_lines.append("Complexity breakdown (Monomer / Dimer / Multi peptide -- only Monomer is in scope "
                      "for a single-chain SMILES regression dataset; non-monomers are logged here and "
                      "excluded downstream in the diff step):")
    for c, cnt in complexity_counter.most_common():
        log_lines.append(f"  {c}: {cnt}")
    log_lines.append("")
    log_lines.append(f"Non-canonical residue codes detected (unusualAminoAcids.modificationType.name), "
                      f"with occurrence counts across peptide entries ({len(unusual_aa_counter)} distinct codes):")
    for code, cnt in unusual_aa_counter.most_common():
        log_lines.append(f"  {code}: {cnt}")
    log_lines.append("")
    log_lines.append(f"Entries with >=1 intrachain bond of a type other than DSB/AMD: {n_with_nonstd_bond}")
    log_lines.append(f"All intrachain bond types observed, with entry counts:")
    for bt, cnt in all_bond_type_counter.most_common():
        log_lines.append(f"  {bt}: {cnt}")
    log_lines.append("")
    log_lines.append(f"Non-DSB/AMD bond types specifically, with entry counts:")
    for bt, cnt in non_std_bond_counter.most_common():
        log_lines.append(f"  {bt}: {cnt}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
