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

qmap.toolkit.train_test_split defaults to post_filtering=True: after clustering
every peptide into train/test, it removes any train-assigned peptide that has a
similarity edge (>= threshold identity) to a test peptide, to guarantee train and
test are genuinely independent. This filter only ever removes from train -- test
is untouched -- and the removed peptides are dropped from the function's return
value entirely, with no indication of which peptides were removed. Earlier
versions of this script wrote train_ids/test_ids straight through and never
reconciled them against the full peptide set, so peptides removed by this filter
silently ended up with no split assignment at all (a QA pass caught 1,414 such
peptides missing from data/split_indices.json). This version explicitly computes
and reassigns any such peptides to the test set -- not train, since filter_out
only flags them for being *too close to test*, so keeping them out of train
preserves the independence guarantee -- and records the reassignment separately
in split_indices.json and in this step's log, so the gap can never again go
unnoticed.
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

POST_FILTERING_NOTE = (
    "qmap.toolkit.train_test_split's post_filtering=True (the library default, used here) "
    "removes any train-assigned peptide with a similarity edge (>= identity_threshold) to a "
    "test peptide, to guarantee train/test independence. This filter only ever removes from "
    "train, never test. The peptide_ids in leakage_filter_reassigned_to_test_peptide_ids are "
    "exactly the ones this filter removed from train; they are added to test_peptide_ids "
    "(also included there) rather than left unassigned or put back in train, since putting "
    "them in test does not violate the independence guarantee the filter exists to enforce."
)


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

    dropped_ids = []
    try:
        from qmap.toolkit import train_test_split
        train_seqs, test_seqs, train_ids, test_ids = train_test_split(
            sequences, peptide_ids,
            threshold=IDENTITY_THRESHOLD, test_size=TEST_SIZE,
            random_state=RANDOM_SEED, verbose=True,
        )
        split_ok = True

        # post_filtering (default True) removes train peptides with an edge to a test
        # peptide, and drops them from both returned lists with no record. Reconcile
        # against the full peptide set and reassign anything missing to test.
        n_train_pre_reassign = len(train_ids)
        n_test_pre_reassign = len(test_ids)
        assigned = set(train_ids) | set(test_ids)
        dropped_ids = sorted(set(peptide_ids) - assigned)
        test_ids = list(test_ids) + dropped_ids

    except Exception as e:
        log_lines.append(f"\nWARNING: qmap.toolkit.train_test_split failed: {e}")
        log_lines.append("Falling back: NO split performed; split_indices.json will be empty. "
                          "Flagged for manual review.")
        train_ids, test_ids = [], []
        n_train_pre_reassign, n_test_pre_reassign = 0, 0
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
        "post_filtering": True,
        "post_filtering_note": POST_FILTERING_NOTE if split_ok else "",
        "leakage_filter_reassigned_to_test_peptide_ids": [str(x) for x in dropped_ids],
        "train_peptide_ids": [str(x) for x in train_ids],
        "test_peptide_ids": [str(x) for x in test_ids],
    }
    with open(SPLIT_JSON, "w") as f:
        json.dump(split_indices, f, indent=2)

    if split_ok:
        n_train_rows = sum(1 for r in final_rows if r["peptide_id"] in train_id_set)
        n_test_rows = sum(1 for r in final_rows if r["peptide_id"] in test_id_set)
        log_lines.append("")
        log_lines.append(f"Split (pre leakage-filter reconciliation): "
                          f"{n_train_pre_reassign} train peptides / {n_test_pre_reassign} test peptides "
                          f"(threshold={IDENTITY_THRESHOLD}, target test_size={TEST_SIZE}, seed={RANDOM_SEED})")
        log_lines.append(f"post_filtering=True removed {len(dropped_ids)} peptides from train for having "
                          f"a >= {IDENTITY_THRESHOLD} identity edge to a test peptide. These are reassigned "
                          f"to test (see leakage_filter_reassigned_to_test_peptide_ids in split_indices.json) "
                          f"rather than left unassigned.")
        log_lines.append(f"Final split: {len(train_ids)} train peptides / {len(test_ids)} test peptides")
        log_lines.append(f"  train rows: {n_train_rows}, test rows: {n_test_rows}")
        log_lines.append(f"Reconciliation check: train + test == unique peptides? "
                          f"{len(train_ids)} + {len(test_ids)} = {len(train_ids) + len(test_ids)} "
                          f"(unique peptides = {len(peptide_ids)}) -> "
                          f"{'OK' if len(train_ids) + len(test_ids) == len(peptide_ids) else 'MISMATCH -- INVESTIGATE'}")
        overlap = train_id_set & test_id_set
        log_lines.append(f"Train/test overlap check: {len(overlap)} peptide_ids in both "
                          f"-> {'OK' if not overlap else 'MISMATCH -- INVESTIGATE'}")

    log_lines.append("")
    log_lines.append(f"Final dataset: {FINAL_CSV}")
    log_lines.append(f"Split indices: {SPLIT_JSON}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
