"""
Section 5: train/test split integrity.

Verifies split_indices.json's train/test peptide_id sets are disjoint, their
union exactly equals the final dataset's unique peptide_id set (no orphans
either direction), sizes match the audit's claimed 11,114/3,376, and every
organism-row in the final dataset has a split assignment consistent with its
peptide (split_level is documented as "unique peptide_id").
"""
from qa_common import QAContext, SectionResult, fmt_int, md_table

EXPECTED_TRAIN = 11114
EXPECTED_TEST = 3376


def run(ctx: QAContext) -> SectionResult:
    follow_ups = []
    narrative = []

    train_ids = set(ctx.split["train_peptide_ids"])
    test_ids = set(ctx.split["test_peptide_ids"])
    final_ids = set(ctx.final["peptide_id"].unique())

    overlap = train_ids & test_ids
    union = train_ids | test_ids
    orphan_in_final_not_split = final_ids - union
    orphan_in_split_not_final = union - final_ids

    n_train, n_test = len(train_ids), len(test_ids)
    size_mismatch = (n_train != EXPECTED_TRAIN) or (n_test != EXPECTED_TEST)

    # row-level consistency: every row for a given peptide_id must fall in exactly one split
    final_train_rows = ctx.final[ctx.final["peptide_id"].isin(train_ids)]
    final_test_rows = ctx.final[ctx.final["peptide_id"].isin(test_ids)]
    row_overlap_peptides = set(final_train_rows["peptide_id"]) & set(final_test_rows["peptide_id"])

    status = "PASS"
    if overlap:
        status = "FAIL"
        follow_ups.append(f"{len(overlap)} peptide_ids appear in BOTH train and test splits -- "
                           f"this is train/test leakage.")
    if orphan_in_final_not_split:
        status = "FAIL"
        follow_ups.append(f"{len(orphan_in_final_not_split)} peptide_ids present in the final "
                           f"dataset have no split assignment at all.")
    if orphan_in_split_not_final:
        status = "FAIL"
        follow_ups.append(f"{len(orphan_in_split_not_final)} peptide_ids appear in the split but "
                           f"not in the final dataset -- stale split, or final assembly dropped "
                           f"peptides after the split was computed.")
    if row_overlap_peptides:
        status = "FAIL"
        follow_ups.append(f"{len(row_overlap_peptides)} peptide_ids have organism-rows that are "
                           f"inconsistent with a clean per-peptide split assignment.")
    if size_mismatch and status == "PASS":
        status = "FLAGGED"
        follow_ups.append(f"Split sizes (train={n_train}, test={n_test}) do not match the audit "
                           f"report's claimed 11,114/3,376 -- may just be a stale audit doc, not "
                           f"a leakage bug, but worth confirming which is current.")

    narrative.append(f"Train set: **{fmt_int(n_train)}** peptide_ids (audit claims {EXPECTED_TRAIN:,}). "
                      f"Test set: **{fmt_int(n_test)}** peptide_ids (audit claims {EXPECTED_TEST:,}). "
                      f"Train/test overlap: **{fmt_int(len(overlap))}** peptide_ids. Final-dataset "
                      f"peptides missing a split assignment: **{fmt_int(len(orphan_in_final_not_split))}**. "
                      f"Split peptide_ids absent from the final dataset: "
                      f"**{fmt_int(len(orphan_in_split_not_final))}**. Peptides with row-level "
                      f"split inconsistency: **{fmt_int(len(row_overlap_peptides))}**.")

    tables = []
    if overlap:
        tables.append(("Sample of train/test overlapping peptide_ids", [{"peptide_id": p} for p in list(overlap)[:20]]))
    if orphan_in_final_not_split:
        tables.append(("Sample of final peptides missing a split assignment", [{"peptide_id": p} for p in list(orphan_in_final_not_split)[:20]]))
    if orphan_in_split_not_final:
        tables.append(("Sample of split peptide_ids absent from final dataset", [{"peptide_id": p} for p in list(orphan_in_split_not_final)[:20]]))

    return SectionResult(
        section_id="5", title="Train/test split integrity", status=status,
        metrics={
            "n_train": n_train, "n_test": n_test, "n_overlap": len(overlap),
            "n_orphan_final_not_split": len(orphan_in_final_not_split),
            "n_orphan_split_not_final": len(orphan_in_split_not_final),
            "n_row_level_inconsistent": len(row_overlap_peptides),
        },
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
