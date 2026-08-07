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
- Confirm the one documented manual-review flag (peptide_id=21052) matches
  the precise, now-corrected understanding of it: QMAP's own pulled copy
  (data/qmap_included.csv) has a leading-whitespace sequence and was never
  meant to be corrected there, while the final dataset's sequence for this
  peptide is sourced independently (from the project's own DBAASP crawl,
  which was already clean at the source) and is expected to be clean. This
  checks BOTH locations so it can actually detect future drift (e.g. if
  QMAP's copy gets cleaned, or the final dataset's copy ever picks up
  whitespace) instead of permanently re-flagging a now-understood non-issue.
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
    # Expected reality (traced 2026-08-07): QMAP's own pulled copy has the whitespace
    # and was never meant to be corrected there; the final dataset sources `sequence`
    # from an entirely different, independent crawl that was already clean. Check both.
    qmap_flag_rows = ctx.qmap_included[ctx.qmap_included["peptide_id"] == "21052"]
    final_flag_rows = ctx.final[ctx.final["peptide_id"] == "21052"]

    qmap_seq = qmap_flag_rows.iloc[0]["sequence"] if len(qmap_flag_rows) > 0 else None
    final_seq = final_flag_rows.iloc[0]["sequence"] if len(final_flag_rows) > 0 else None

    qmap_has_whitespace = qmap_seq is not None and qmap_seq != qmap_seq.strip()
    final_is_clean = final_seq is not None and final_seq == final_seq.strip()

    if qmap_seq is None or final_seq is None:
        flag_status = "missing_from_one_or_both_sources"
    elif qmap_has_whitespace and final_is_clean:
        flag_status = "as_expected"  # qmap copy dirty (undocumented-fix-worthy but out of scope), final clean
    elif not qmap_has_whitespace and final_is_clean:
        flag_status = "qmap_copy_unexpectedly_cleaned"
    elif not final_is_clean:
        flag_status = "final_dataset_unexpectedly_dirty"
    else:
        flag_status = "unexpected_state"
    flag_sequence_sample = f"qmap_included={qmap_seq!r}, final={final_seq!r}"

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
    if flag_status == "missing_from_one_or_both_sources":
        status = "FAIL"
        follow_ups.append("peptide_id=21052 is missing from data/qmap_included.csv and/or the "
                           "final dataset entirely -- the documented caveat can no longer be "
                           "traced at all; investigate whether this peptide was dropped somewhere.")
    elif flag_status == "final_dataset_unexpectedly_dirty":
        status = "FAIL"
        follow_ups.append("peptide_id=21052's sequence in the FINAL dataset now has whitespace -- "
                           "this is a real data-quality regression (the deliverable dataset should "
                           "never carry this artifact), not just a stale caveat.")
    elif flag_status == "qmap_copy_unexpectedly_cleaned":
        if status == "PASS":
            status = "FLAGGED"
        follow_ups.append("data/qmap_included.csv's copy of peptide_id=21052's sequence no longer "
                           "has the documented whitespace -- QMAP's source pull may have changed; "
                           "harmless to the final dataset either way, but worth a note update.")
    elif flag_status == "unexpected_state":
        if status == "PASS":
            status = "FLAGGED"
        follow_ups.append("peptide_id=21052's sequences in qmap_included.csv and the final dataset "
                           "are in a combination not anticipated by the documented caveat -- see "
                           "flag_sequence_sample for the raw values.")
    # flag_status == "as_expected": qmap_included.csv retains the whitespace (undocumented as a
    # separate fix-worthy issue in that file, but out of scope here) and the final dataset is
    # clean, sourced independently -- exactly the corrected understanding documented in
    # reports/curation_audit.md as of 2026-08-07. No status change.

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
    narrative.append(f"**Manual-review flag (peptide_id=21052):** status = `{flag_status}` "
                      f"({flag_sequence_sample}). Expected: qmap_included.csv retains the "
                      f"undocumented-elsewhere whitespace, final dataset is clean because it's "
                      f"sourced from an independently-crawled, already-clean copy -- not because "
                      f"anything was corrected mid-pipeline.")

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
