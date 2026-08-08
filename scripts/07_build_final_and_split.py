"""
Step 6: finalize the combined dataset (QMAP baseline + recovered, already assay-
filtered and unit-standardized by step 5) into the requested schema, and apply a
homology-aware train/test split using QMAP's own splitting utility
(qmap.toolkit.train_test_split: sequence-identity graph + Leiden community
detection, default 60% identity threshold), so the split methodology matches the
paper this dataset follows.

The split is computed at the peptide (unique sequence) level -- not per output
row -- so that all (organism) rows sharing a peptide stay in the same side of the
split; splitting at the row level would leak near-identical sequences across
train/test through their organism-duplicated rows, defeating the point of a
homology-aware split.
"""
import sys
import os
import csv
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

STEP5_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "step5_standardized_mic.csv")
FINAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "final_mic_regression_dataset.csv")
SPLIT_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "split_indices.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "step6_final_split_log.txt")

FINAL_FIELDNAMES = [
    "peptide_id", "sequence", "smiles", "organism", "ncbi_taxon_id_if_available",
    "mic_value_uM", "mic_type", "source", "has_noncanonical", "bond_type", "molecular_weight",
]

RANDOM_SEED = 42
IDENTITY_THRESHOLD = 0.60
TEST_SIZE = 0.2


def main():
    with open(STEP5_CSV) as f:
        rows = list(csv.DictReader(f))

    final_rows = []
    for r in rows:
        final_rows.append({
            "peptide_id": r["peptide_id"],
            "sequence": r["sequence"],
            "smiles": r["smiles"],
            "organism": r["organism"],
            "ncbi_taxon_id_if_available": r["ncbi_taxon_id"],
            "mic_value_uM": r["mic_value_uM"],
            "mic_type": r["mic_type"],
            "source": "qmap_original" if r["source"] == "qmap_original" else "recovered",
            "has_noncanonical": r["has_noncanonical"],
            "bond_type": r["bond_type"],
            "molecular_weight": r["molecular_weight"],
        })

    os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(final_rows)

    # --- unique peptide-level sequences for the split ---
    unique_peptides = {}
    for r in final_rows:
        pid = r["peptide_id"]
        if pid not in unique_peptides:
            unique_peptides[pid] = r["sequence"]
    peptide_ids = list(unique_peptides.keys())
    sequences = [unique_peptides[pid] for pid in peptide_ids]

    log_lines = []
    log_lines.append("=== Step 6: final dataset assembly + homology-aware split ===")
    log_lines.append(f"Final table rows (peptide x organism MIC pairs): {len(final_rows)}")
    log_lines.append(f"Unique peptides: {len(peptide_ids)}")
    n_qmap = sum(1 for r in final_rows if r["source"] == "qmap_original")
    n_recovered = sum(1 for r in final_rows if r["source"] == "recovered")
    log_lines.append(f"  rows from source=qmap_original: {n_qmap}")
    log_lines.append(f"  rows from source=recovered: {n_recovered}")
    uniq_qmap = len({r["peptide_id"] for r in final_rows if r["source"] == "qmap_original"})
    uniq_recovered = len({r["peptide_id"] for r in final_rows if r["source"] == "recovered"})
    log_lines.append(f"  unique peptides source=qmap_original: {uniq_qmap}")
    log_lines.append(f"  unique peptides source=recovered: {uniq_recovered}")

    try:
        from qmap.toolkit import train_test_split
        train_seqs, test_seqs, train_ids, test_ids = train_test_split(
            sequences, peptide_ids,
            threshold=IDENTITY_THRESHOLD, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, verbose=True,
        )
        split_ok = True
    except Exception as e:
        log_lines.append(f"\nWARNING: qmap.toolkit.train_test_split failed: {e}")
        log_lines.append("Falling back: NO split performed; split_indices.json will be empty. "
                          "Flagged for manual review.")
        train_ids, test_ids = [], []
        split_ok = False

    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    split_indices = {
        "method": "qmap.toolkit.train_test_split (sequence-identity graph + Leiden community "
                  "detection)" if split_ok else "FAILED -- not computed",
        "identity_threshold": IDENTITY_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_SEED,
        "split_level": "unique peptide_id (all organism rows for a peptide share its split assignment)",
        "train_peptide_ids": [str(x) for x in train_ids],
        "test_peptide_ids": [str(x) for x in test_ids],
    }
    with open(SPLIT_JSON, "w") as f:
        json.dump(split_indices, f, indent=2)

    if split_ok:
        n_train_rows = sum(1 for r in final_rows if r["peptide_id"] in train_id_set)
        n_test_rows = sum(1 for r in final_rows if r["peptide_id"] in test_id_set)
        log_lines.append("")
        log_lines.append(f"Split: {len(train_ids)} train peptides / {len(test_ids)} test peptides "
                          f"(threshold={IDENTITY_THRESHOLD}, target test_size={TEST_SIZE}, seed={RANDOM_SEED})")
        log_lines.append(f"  train rows: {n_train_rows}, test rows: {n_test_rows}")

    log_lines.append("")
    log_lines.append(f"Final dataset: {FINAL_CSV}")
    log_lines.append(f"Split indices: {SPLIT_JSON}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
