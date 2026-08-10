"""
Step 1: Load QMAP's included dataset (anthol42/qmap_benchmark_2025, dbaasp.json)
and flatten it to a per (peptide, organism) MIC-regression table.

Source format (verified by reading QMAP's own build_dataset.py, which produced
this exact file):
    {
      "id": int,                       # DBAASP peptide id
      "sequence": str,                 # QMAP "common_sequence": DBAASP raw sequence
                                        # with ORN/D-ORN/DAB/D-DAB substituted to O/o/B/b;
                                        # case preserved for D-amino acids; "X" may remain
                                        # for other unresolved non-canonical residues IF a
                                        # native SMILES was available (QMAP's inclusion rule).
      "smiles": [str, ...],            # native DBAASP SMILES field, 0+ entries
      "nterminal": "ACT" | None,
      "cterminal": "AMD" | None,
      "bonds": [[start, end, type], ...],   # type in {"DSB", "AMD"} only (QMAP filter)
      "targets": {species: [min_uM, max_uM, mean_uM_IQR_cleaned], ...},  # MIC, bacteria only
      "hemolytic_hc50": [min, max, mean] | None
    }

We do NOT re-derive any of QMAP's filtering/unit-conversion here -- this step is
purely "load QMAP's baseline as-is" per the task spec. Re-derivation of DBAASP's
raw data happens independently in the DBAASP crawl (01_crawl_dbaasp.py /
02_parse_dbaasp_raw.py) so the diff in step 3 is a genuine independent check.
"""
import json
import os
import re
import csv
from collections import Counter

HF_JSON = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "qmap_hf", "dbaasp.json")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "qmap_included.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "step1_qmap_pull_log.txt")

CANONICAL_L = set("ACDEFGHIKLMNPQRSTVWY")


def has_noncanonical(sequence: str) -> bool:
    """True if sequence contains any character outside the 20 canonical
    uppercase (L-form) amino acid letters -- i.e. lowercase (D-form),
    O/o (Orn/D-Orn), B/b (Dab/D-Dab), X (other unresolved), or anything else."""
    return any(ch not in CANONICAL_L for ch in sequence)


def noncanonical_kind(sequence: str) -> str:
    """Classify the *reason* a sequence is flagged non-canonical, for the
    residue-type breakdown requested in the task."""
    reasons = set()
    for ch in sequence:
        if ch in CANONICAL_L:
            continue
        if ch in ("O", "o"):
            reasons.add("ornithine")
        elif ch in ("B", "b"):
            reasons.add("dab")
        elif ch == "X":
            reasons.add("other_unresolved(has_smiles)")
        elif ch.isalpha() and ch.islower():
            reasons.add("D-canonical-residue")
        else:
            reasons.add(f"other_char:{ch}")
    if not reasons:
        return "canonical"
    return "+".join(sorted(reasons))


def main():
    with open(HF_JSON) as f:
        data = json.load(f)

    rows = []
    n_with_smiles = 0
    n_multi_smiles = 0
    organisms = Counter()
    noncanon_kind_counter = Counter()
    bond_type_counter = Counter()
    n_peptides_noncanon = 0
    n_target_rows_total = 0

    for entry in data:
        pid = entry["id"]
        seq = entry["sequence"]
        smiles_list = entry.get("smiles") or []
        smiles_joined = "|".join(smiles_list)
        if smiles_list:
            n_with_smiles += 1
        if len(smiles_list) > 1:
            n_multi_smiles += 1

        bonds = entry.get("bonds") or []
        bond_types = sorted({b[2] for b in bonds}) if bonds else []
        bond_type_str = "|".join(bond_types) if bond_types else "none"
        for bt in (bond_types or ["none"]):
            bond_type_counter[bt] += 1

        nc = has_noncanonical(seq)
        kind = noncanonical_kind(seq)
        if nc:
            n_peptides_noncanon += 1
        noncanon_kind_counter[kind] += 1

        targets = entry.get("targets") or {}
        for organism, (min_v, max_v, mean_v) in targets.items():
            organisms[organism] += 1
            n_target_rows_total += 1
            rows.append({
                "peptide_id": pid,
                "sequence": seq,
                "smiles": smiles_joined,
                "organism": organism,
                "mic_value_uM": mean_v,
                "mic_min_uM": min_v,
                "mic_max_uM": max_v,
                "source_db": "DBAASP_via_QMAP",
                "has_noncanonical": nc,
                "bond_type": bond_type_str,
                "nterminal": entry.get("nterminal"),
                "cterminal": entry.get("cterminal"),
            })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_peptides = len(data)
    n_peptides_with_any_target = sum(1 for e in data if e.get("targets"))
    n_peptides_no_target = n_peptides - n_peptides_with_any_target

    flagged = []
    for e in data:
        if e["sequence"] != e["sequence"].strip():
            flagged.append(f"peptide_id={e['id']}: sequence has leading/trailing whitespace: {e['sequence']!r}")

    log_lines = []
    log_lines.append("=== Step 1: QMAP included dataset pull ===")
    log_lines.append(f"Source: huggingface anthol42/qmap_benchmark_2025 / dbaasp.json")
    log_lines.append(f"Total peptide records in QMAP dataset: {n_peptides}")
    log_lines.append(f"Peptides with >=1 bacterial MIC target (species) recorded: {n_peptides_with_any_target}")
    log_lines.append(f"Peptides with native DBAASP SMILES present: {n_with_smiles} ({100*n_with_smiles/n_peptides:.1f}%)")
    log_lines.append(f"  - of which have >1 SMILES candidate listed: {n_multi_smiles}")
    log_lines.append(f"Peptides flagged has_noncanonical=True: {n_peptides_noncanon} ({100*n_peptides_noncanon/n_peptides:.1f}%)")
    log_lines.append(f"Peptides canonical-only: {n_peptides - n_peptides_noncanon}")
    log_lines.append("")
    log_lines.append("Non-canonical breakdown (peptide-level, by residue kind combination present):")
    for kind, cnt in noncanon_kind_counter.most_common():
        log_lines.append(f"  {kind}: {cnt}")
    log_lines.append("")
    log_lines.append("Bond type counts (peptide-level, DSB/AMD only per QMAP filter):")
    for bt, cnt in bond_type_counter.most_common():
        log_lines.append(f"  {bt}: {cnt}")
    log_lines.append("")
    log_lines.append(f"MIC unit: uM for all 'targets' entries (QMAP's target_base.py converts "
                      f"ug/mL -> uM via computed molecular weight before this file is produced; "
                      f"confirmed by reading QMAP source, no unit field is stored in this flattened "
                      f"export so this is a methodology-level confirmation, not a per-row field).")
    log_lines.append(f"Unique organisms across all (peptide, organism) MIC rows: {len(organisms)}")
    log_lines.append(f"Total (peptide, organism) MIC rows written to {OUT_CSV}: {n_target_rows_total}")
    log_lines.append(f"Peptides with zero bacterial MIC targets (present in QMAP's 18,033 but "
                      f"contribute 0 rows to the flattened regression table -- e.g. hemolysis-only "
                      f"entries or species that didn't pass QMAP's Bacteria/MIC/unit filters): "
                      f"{n_peptides_no_target}")
    log_lines.append("")
    if flagged:
        log_lines.append(f"MANUAL REVIEW FLAGS ({len(flagged)}):")
        for line in flagged:
            log_lines.append(f"  {line}")
        log_lines.append("")
    log_lines.append("Top 20 organisms by row count:")
    for org, cnt in organisms.most_common(20):
        log_lines.append(f"  {org}: {cnt}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
