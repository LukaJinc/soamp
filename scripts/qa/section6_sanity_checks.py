"""
Section 6: row-level sanity checks and documented-caveat verification.

- Duplicate (peptide_id, organism) pairs in the final dataset (grouping in
  Step 5 should have collapsed these to one row each).
- mic_value_uM sanity: no non-positive, NaN, or infinite values.
- Self-consistency between step5's mic_type label and its own
  n_raw_measurements column (averaged iff n_raw_measurements > 1), and a
  cross-check of the final dataset's mic_type breakdown against the audit
  report's claimed Final-dataset-composition numbers. IQR outlier-point
  counts are NOT independently re-derived here (that requires reimplementing
  the range/censoring value parser, out of scope per qa_common.py's
  disclosed exceptions) -- reported as an audit-claimed figure only, not
  independently verified.
- Confirm the one documented manual-review flag (peptide_id=21052, leading
  whitespace in the raw sequence) is traceable in the final data as
  described, not silently dropped or silently fixed without note.
"""
import math
from qa_common import QAContext, SectionResult, fmt_int, md_table

AUDIT_MIC_TYPE = {"exact": 38808, "censored": 18323, "averaged": 15456}
AUDIT_GROUPS_MULTI = 15456


def run(ctx: QAContext) -> SectionResult:
    follow_ups = []
    narrative = []

    # --- duplicates ---
    dup_mask = ctx.final.duplicated(subset=["peptide_id", "organism"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    dup_examples = ctx.final[dup_mask][["peptide_id", "organism"]].drop_duplicates().head(20).to_dict("records")

    # --- value sanity ---
    vals = ctx.final["mic_value_uM"]
    n_nonpositive = int(((vals <= 0) | vals.isna()).sum())
    n_infinite = int(vals.apply(lambda x: isinstance(x, float) and math.isinf(x)).sum())
    bad_value_examples = ctx.final[(vals <= 0) | vals.isna()][["peptide_id", "organism", "mic_value_uM"]].head(20).to_dict("records")

    # --- mic_type self-consistency (step5: averaged label iff n_raw_measurements > 1) ---
    step5 = ctx.step5
    inconsistent = step5[
        ((step5["n_raw_measurements"] > 1) & (step5["mic_type"] != "averaged")) |
        ((step5["n_raw_measurements"] == 1) & (step5["mic_type"] == "averaged"))
    ]
    n_mic_type_inconsistent = len(inconsistent)

    n_groups_multi_recomputed = int((step5["n_raw_measurements"] > 1).sum())
    groups_multi_mismatch = n_groups_multi_recomputed != AUDIT_GROUPS_MULTI

    final_mic_type_counts = ctx.final["mic_type"].value_counts().to_dict()
    mic_type_mismatches = {k: (v, final_mic_type_counts.get(k)) for k, v in AUDIT_MIC_TYPE.items()
                            if final_mic_type_counts.get(k) != v}

    # --- documented manual-review flag: peptide_id=21052 ---
    flag_rows = ctx.final[ctx.final["peptide_id"] == "21052"]
    flag_status = "not_present"
    flag_sequence_sample = None
    if len(flag_rows) > 0:
        seq = flag_rows.iloc[0]["sequence"]
        flag_sequence_sample = repr(seq)
        flag_status = "present_with_whitespace" if seq != seq.strip() else "present_whitespace_stripped"

    status = "PASS"
    if n_dup_rows:
        status = "FAIL"
        follow_ups.append(f"{n_dup_rows} rows in the final dataset share a duplicate "
                           f"(peptide_id, organism) key -- Step 5's grouping should guarantee "
                           f"uniqueness; this indicates two groups were not merged.")
    if n_nonpositive or n_infinite:
        status = "FAIL"
        follow_ups.append(f"{n_nonpositive} rows have a non-positive or missing mic_value_uM and "
                           f"{n_infinite} have an infinite value -- both are physically invalid for "
                           f"a concentration.")
    if n_mic_type_inconsistent:
        status = "FAIL"
        follow_ups.append(f"{n_mic_type_inconsistent} step5 rows have a mic_type label inconsistent "
                           f"with their own n_raw_measurements count (averaged should mean "
                           f"n_raw_measurements > 1, and only n_raw_measurements > 1).")
    if mic_type_mismatches or groups_multi_mismatch:
        if status == "PASS":
            status = "FLAGGED"
        follow_ups.append("Final dataset's mic_type breakdown and/or averaged-group count differs "
                           "from the audit report's claimed Final-dataset-composition numbers -- "
                           "see table (may just mean the audit doc is stale relative to the data "
                           "on disk).")
    if flag_status == "not_present":
        if status == "PASS":
            status = "FLAGGED"
        follow_ups.append("Documented manual-review flag peptide_id=21052 (leading-whitespace "
                           "sequence) is not present in the final dataset at all -- was it dropped "
                           "somewhere in the pipeline, and if so is that intentional?")
    elif flag_status == "present_whitespace_stripped":
        if status == "PASS":
            status = "FLAGGED"
        follow_ups.append("peptide_id=21052's sequence no longer has the documented leading "
                           "whitespace -- it appears to have been silently corrected somewhere in "
                           "the pipeline without updating the audit report's caveat.")

    narrative.append(f"**Duplicates:** {fmt_int(n_dup_rows)} rows share a duplicate "
                      f"(peptide_id, organism) key.")
    narrative.append(f"**Value sanity:** {fmt_int(n_nonpositive)} non-positive/missing mic_value_uM "
                      f"rows, {fmt_int(n_infinite)} infinite-value rows.")
    narrative.append(f"**mic_type self-consistency:** {fmt_int(n_mic_type_inconsistent)} step5 rows "
                      f"have a mic_type label inconsistent with their own n_raw_measurements count. "
                      f"Independently recomputed averaged-group count (n_raw_measurements > 1): "
                      f"**{fmt_int(n_groups_multi_recomputed)}** (audit claims {AUDIT_GROUPS_MULTI:,}).")
    narrative.append(f"**mic_type breakdown vs audit:** {mic_type_mismatches if mic_type_mismatches else 'matches exactly'}.")
    narrative.append(f"**IQR outlier-point counts** (audit claims 2,236 groups / 4,062 points) are "
                      f"reported as-is from the audit, not independently re-derived here -- doing so "
                      f"would require reimplementing `parse_activity`'s range/censoring string "
                      f"parser, which is out of scope for this independent pass (disclosed in "
                      f"`qa_common.py`).")
    narrative.append(f"**Manual-review flag (peptide_id=21052):** status = `{flag_status}`"
                      + (f", sequence = {flag_sequence_sample}" if flag_sequence_sample else "") + ".")

    tables = []
    if dup_examples:
        tables.append(("Duplicate (peptide_id, organism) keys (sample, up to 20)", dup_examples))
    if bad_value_examples:
        tables.append(("Invalid mic_value_uM rows (sample, up to 20)", bad_value_examples))
    if mic_type_mismatches:
        tables.append(("mic_type breakdown: audit-claimed vs actual", [
            {"mic_type": k, "audit_claimed": v[0], "actual": v[1]} for k, v in mic_type_mismatches.items()
        ]))

    return SectionResult(
        section_id="6", title="Row-level sanity & documented-caveat checks", status=status,
        metrics={
            "n_duplicate_rows": n_dup_rows, "n_nonpositive_or_missing": n_nonpositive,
            "n_infinite": n_infinite, "n_mic_type_inconsistent": n_mic_type_inconsistent,
            "n_groups_multi_recomputed": n_groups_multi_recomputed,
            "flag_21052_status": flag_status,
        },
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
