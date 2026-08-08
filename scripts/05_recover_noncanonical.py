"""
Step 4: recover eligible peptides from the diff's "excluded from QMAP" buckets
(b1-b5 in data/dbaasp_vs_qmap_diff.csv).

Recovery policy per bucket:
  b1_excluded_non_monomer          -- NOT recoverable: dimers/multi-peptide complexes
                                       are out of scope for a single-chain SMILES
                                       regression dataset. Logged, not silently dropped.
  b2/b3/b4/b5 (terminus/residue/bond/unexplained):
     - if DBAASP provides a native SMILES: used as-is (no re-derivation) -- this is
       the direct "bucket (b) with SMILES available" recovery the task asks for.
     - else: attempt generation via p2smi (smiles_gen.generate_smiles), covering
       linear peptides with resolvable residues, disulfide (SS) and head-to-tail
       (HT) cyclization. Anything p2smi/residue_map can't resolve is logged as
       unconvertible with the specific reason -- never silently dropped, never
       approximated with a nearest-analog substitution.

Output: data/recovered_peptides.csv (peptide_id, sequence, smiles, recovered_via,
recovery_category, nterminus, cterminus, bond_types, has_noncanonical), plus
reports/step4_recovery_log.txt with full before/after counts by reason.
"""
import sys
import os
import json
import csv
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from parse_dbaasp import DBAASPPeptide
from smiles_gen import generate_smiles

RAW_JSONL = os.path.join(os.path.dirname(__file__), "..", ".cache", "dbaasp_raw.jsonl")
DIFF_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "dbaasp_vs_qmap_diff.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "recovered_peptides.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "step4_recovery_log.txt")

BUCKET_TO_CATEGORY = {
    "b2_excluded_unsupported_terminus": "terminus_based",
    "b3_excluded_residue_no_smiles": "residue_based",
    "b4_excluded_nonstandard_bond": "bond_based",
    "b5_excluded_unexplained": "unexplained_temporal_drift",
}


def main():
    raw_peptides = {}
    with open(RAW_JSONL) as f:
        for line in f:
            raw = json.loads(line)
            raw_peptides[raw["id"]] = DBAASPPeptide(raw)

    diff_rows = []
    with open(DIFF_CSV) as f:
        for row in csv.DictReader(f):
            diff_rows.append(row)

    recovered = []
    unconvertible = []
    n_not_recoverable_scope = 0
    by_bucket_total = Counter()
    by_bucket_recovered = Counter()
    recovered_via_counter = Counter()
    unconvertible_reason_counter = Counter()

    for row in diff_rows:
        bucket = row["bucket"]
        if bucket in ("in_both", "c_in_qmap_not_in_raw_pull"):
            continue  # not part of the recovery pool

        by_bucket_total[bucket] += 1
        pid = int(row["peptide_id"])

        if bucket == "b1_excluded_non_monomer":
            n_not_recoverable_scope += 1
            unconvertible.append({
                "peptide_id": pid, "bucket": bucket, "category": "out_of_scope",
                "reason": "non-monomer complexity (dimer/multi-peptide) -- not a single-chain "
                          "SMILES peptide, out of scope for this dataset",
            })
            continue

        p = raw_peptides.get(pid)
        if p is None:
            continue  # shouldn't happen: diff was built from the same raw pull
        category = BUCKET_TO_CATEGORY[bucket]

        if p.has_native_smiles:
            smiles_candidates = p.native_smiles_list
            recovered.append({
                "peptide_id": pid, "sequence": p.sequence, "smiles": smiles_candidates[0],
                "recovered_via": "native_dbaasp_smiles", "recovery_category": category,
                "nterminus": p.nterminus, "cterminus": p.cterminus,
                "bond_types": "|".join(p.bond_types), "has_noncanonical": p.has_noncanonical,
                "n_smiles_candidates": len(smiles_candidates),
            })
            recovered_via_counter["native_dbaasp_smiles"] += 1
            by_bucket_recovered[bucket] += 1
            continue

        result = generate_smiles(p.sequence, p.unusual_amino_acids, p.intrachain_bonds,
                                   p.nterminus, p.cterminus)
        if result["convertible"]:
            recovered.append({
                "peptide_id": pid, "sequence": p.sequence, "smiles": result["smiles"],
                "recovered_via": f"p2smi_generated({result['constraint_used']})",
                "recovery_category": category,
                "nterminus": p.nterminus, "cterminus": p.cterminus,
                "bond_types": "|".join(p.bond_types), "has_noncanonical": p.has_noncanonical,
                "n_smiles_candidates": 1,
            })
            recovered_via_counter[f"p2smi_generated({result['constraint_used']})"] += 1
            by_bucket_recovered[bucket] += 1
        else:
            unconvertible.append({
                "peptide_id": pid, "bucket": bucket, "category": category,
                "reason": result["reason"],
            })
            unconvertible_reason_counter[result["reason"].split(":")[0].split("(")[0].strip()] += 1

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(recovered[0].keys()))
        writer.writeheader()
        writer.writerows(recovered)

    unconv_path = os.path.join(os.path.dirname(__file__), "..", "data", "unconvertible_peptides.csv")
    with open(unconv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unconvertible[0].keys()))
        writer.writeheader()
        writer.writerows(unconvertible)

    log_lines = []
    log_lines.append("=== Step 4: recovery of non-canonical/cyclic peptides excluded from QMAP ===")
    log_lines.append(f"Total candidate entries considered (buckets b1-b5): {sum(by_bucket_total.values())}")
    log_lines.append(f"  of which out of scope (non-monomer, not attempted): {n_not_recoverable_scope}")
    log_lines.append(f"Recovered: {len(recovered)}")
    log_lines.append(f"Unconvertible (logged, excluded): {len(unconvertible)}")
    log_lines.append("")
    log_lines.append("Recovered by source:")
    for k, cnt in recovered_via_counter.most_common():
        log_lines.append(f"  {k}: {cnt}")
    log_lines.append("")
    log_lines.append("Recovered / attempted by bucket (recovery category):")
    for b in sorted(by_bucket_total):
        cat = BUCKET_TO_CATEGORY.get(b, "out_of_scope")
        log_lines.append(f"  {b} ({cat}): {by_bucket_recovered[b]} recovered / {by_bucket_total[b]} total")
    log_lines.append("")
    log_lines.append("Unconvertible reason summary (top-level, truncated):")
    for reason, cnt in unconvertible_reason_counter.most_common(30):
        log_lines.append(f"  {reason}: {cnt}")
    log_lines.append("")
    log_lines.append(f"Full recovered list: {OUT_CSV}")
    log_lines.append(f"Full unconvertible list (with per-entry reason): {unconv_path}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
