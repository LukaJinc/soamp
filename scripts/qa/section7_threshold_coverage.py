"""
Section 7: threshold table structural validity + labeling-stage correctness.

Independently re-derives, from the raw CSVs (not by importing
src/soamp/common/thresholds.py or src/soamp/labeling/labels.py -- same
"independent auditor" posture as the rest of scripts/qa/, see
qa_common.py's module docstring), both:
  - which breakpoint each final-dataset row *should* resolve to (species
    match, else genus match, else none), and
  - what label that row *should* get: reduce mic_type to a plausible-value
    interval (a point [v, v] for exact/averaged, [recovered_min, recovered_max]
    for censored) and compare against the organism's dual
    active_threshold_uM/inactive_threshold_uM breakpoint, then compares
    against data/mic_activity_labels.csv's own columns. A mismatch here is
    a real bug in the labeling pipeline, not a stale-doc issue.

If the labeling stage hasn't been run yet (data/mic_activity_labels.csv
etc. don't exist), this section reports FLAGGED with an explanatory note
rather than failing -- the labeling stage is optional/separate from the
curation pipeline this QA tool otherwise audits.
"""
import math

import pandas as pd
from qa_common import QAContext, SectionResult, fmt_int, md_table

TOP_N_MISSING = 20


def run(ctx: QAContext) -> SectionResult:
    if ctx.threshold_table is None or ctx.censor_direction is None or ctx.labels is None:
        missing = [name for name, df in [
            ("config/thresholds/organism_thresholds.csv", ctx.threshold_table),
            ("data/mic_censor_direction.csv", ctx.censor_direction),
            ("data/mic_activity_labels.csv", ctx.labels),
        ] if df is None]
        return SectionResult(
            section_id="7", title="Threshold coverage & labeling correctness", status="FLAGGED",
            narrative=["Labeling stage has not been run yet -- skipping."],
            follow_ups=[f"Missing: {m}" for m in missing] +
                       ["Run pipeline/labeling/build_threshold_template.py, "
                        "01_recover_censor_direction.py, and 02_binarize_mic_labels.py, "
                        "then re-run QA."],
        )

    follow_ups = []
    narrative = []

    # --- 1. Structural validity of organism_thresholds.csv ---
    tt = ctx.threshold_table
    dup_mask = tt.duplicated(subset=["level", "match_key"], keep=False)
    n_dup_keys = int(dup_mask.sum())
    dup_examples = tt[dup_mask][["level", "match_key"]].drop_duplicates().head(20).to_dict("records")

    bad_level_mask = ~tt["level"].isin(["species", "genus"])
    n_bad_level = int(bad_level_mask.sum())

    def _is_blank(v):
        return v is None or (isinstance(v, float) and math.isnan(v)) or v == ""

    def _parses_as_float(v):
        if _is_blank(v):
            return True  # blank -- legitimately unfilled, not an error
        try:
            float(v)
            return True
        except (ValueError, TypeError):
            return False

    bad_active_mask = ~tt["active_threshold_uM"].apply(_parses_as_float)
    bad_inactive_mask = ~tt["inactive_threshold_uM"].apply(_parses_as_float)
    n_bad_threshold = int((bad_active_mask | bad_inactive_mask).sum())

    partial_fill_mask = tt["active_threshold_uM"].apply(lambda v: not _is_blank(v)) \
        != tt["inactive_threshold_uM"].apply(lambda v: not _is_blank(v))
    n_partial_fill = int(partial_fill_mask.sum())
    partial_fill_examples = tt[partial_fill_mask][["level", "match_key"]].head(20).to_dict("records")

    both_filled_mask = (~bad_active_mask & ~bad_inactive_mask
                         & tt["active_threshold_uM"].apply(lambda v: not _is_blank(v))
                         & tt["inactive_threshold_uM"].apply(lambda v: not _is_blank(v)))
    # pd.to_numeric(errors="coerce") rather than .astype(float): the column may
    # contain non-numeric garbage outside both_filled_mask (already counted in
    # n_bad_threshold), which would otherwise raise before the mask is applied.
    active_numeric = pd.to_numeric(tt["active_threshold_uM"], errors="coerce")
    inactive_numeric = pd.to_numeric(tt["inactive_threshold_uM"], errors="coerce")
    reversed_mask = both_filled_mask & (active_numeric > inactive_numeric)
    n_reversed = int(reversed_mask.sum())
    reversed_examples = tt[reversed_mask][
        ["level", "match_key", "active_threshold_uM", "inactive_threshold_uM"]
    ].head(20).to_dict("records")

    # --- independent species/genus threshold dicts (not importing soamp.common.thresholds) ---
    filled = tt[both_filled_mask]
    species_thresh = {row["match_key"]: (float(row["active_threshold_uM"]), float(row["inactive_threshold_uM"]))
                       for _, row in filled[filled["level"] == "species"].iterrows()}
    genus_thresh = {row["match_key"]: (float(row["active_threshold_uM"]), float(row["inactive_threshold_uM"]))
                     for _, row in filled[filled["level"] == "genus"].iterrows()}

    def _lookup(organism: str):
        if organism in species_thresh:
            return species_thresh[organism], "species"
        genus = organism.split(" ", 1)[0] if organism else ""
        if genus in genus_thresh:
            return genus_thresh[genus], "genus"
        return None, "none"

    # --- independent censoring bounds dict ---
    cd = ctx.censor_direction
    bounds = {(row["peptide_id"], row["organism"]): (row["recovered_min_uM"], row["recovered_max_uM"])
              for _, row in cd.iterrows()}

    def _expected_label(mic_value_uM, mic_type, breakpoint, pid, organism):
        if breakpoint is None:
            return "unlabeled"
        active_uM, inactive_uM = breakpoint
        if mic_type in ("exact", "averaged"):
            lo = hi = mic_value_uM
        elif mic_type == "censored":
            lo, hi = bounds.get((pid, organism), (0.0, float("inf")))
        else:
            return "unknown_mic_type"
        if hi <= active_uM:
            return "active"
        if lo >= inactive_uM:
            return "inactive"
        return "uncertain"

    # --- 2. cross-check every row in data/mic_activity_labels.csv ---
    labels = ctx.labels
    n_total = len(labels)
    match_level_mismatches = []
    label_mismatches = []
    match_level_counts = {"species": 0, "genus": 0, "none": 0}
    label_counts = {}
    total_level_mismatches = 0
    total_label_mismatches = 0

    for _, row in labels.iterrows():
        pid, organism = row["peptide_id"], row["organism"]
        mic_value_uM, mic_type = row["mic_value_uM"], row["mic_type"]
        expected_breakpoint, expected_level = _lookup(organism)
        match_level_counts[expected_level] = match_level_counts.get(expected_level, 0) + 1

        if expected_level != row["threshold_match_level"]:
            total_level_mismatches += 1
            if len(match_level_mismatches) < 20:
                match_level_mismatches.append({
                    "peptide_id": pid, "organism": organism,
                    "expected_level": expected_level, "actual_level": row["threshold_match_level"],
                })

        expected_label = _expected_label(mic_value_uM, mic_type, expected_breakpoint, pid, organism)
        actual_label = row["label"]
        label_counts[actual_label] = label_counts.get(actual_label, 0) + 1
        if expected_label != actual_label:
            total_label_mismatches += 1
            if len(label_mismatches) < 20:
                label_mismatches.append({
                    "peptide_id": pid, "organism": organism, "mic_type": mic_type,
                    "expected_label": expected_label, "actual_label": actual_label,
                })

    n_covered = n_total - match_level_counts.get("none", 0)
    coverage_pct = 100 * n_covered / n_total if n_total else 0.0

    # --- 3. censor-direction recovery completeness ---
    n_censored_final = int((ctx.final["mic_type"] == "censored").sum())
    n_censored_recovered = int((cd["recovery_status"] == "recovered").sum())
    n_censored_rows_in_cd = len(cd)

    # --- 4. punch list: highest-impact species still missing a threshold ---
    filled_species_keys = set(species_thresh.keys())
    missing_species = tt[(tt["level"] == "species") & (~tt["match_key"].isin(filled_species_keys))]
    missing_species = missing_species.sort_values("n_dataset_rows", ascending=False)
    punch_list = missing_species.head(TOP_N_MISSING)[["match_key", "n_dataset_rows"]].to_dict("records")

    status = "PASS"
    if (total_level_mismatches or total_label_mismatches or n_dup_keys or n_bad_level
            or n_bad_threshold or n_partial_fill or n_reversed):
        status = "FAIL"
        if n_dup_keys:
            follow_ups.append(f"{n_dup_keys} duplicate (level, match_key) rows in "
                               f"organism_thresholds.csv -- lookup is ambiguous for these keys.")
        if n_bad_level:
            follow_ups.append(f"{n_bad_level} rows have a level other than 'species'/'genus'.")
        if n_bad_threshold:
            follow_ups.append(f"{n_bad_threshold} rows have a non-numeric, non-blank threshold value.")
        if n_partial_fill:
            follow_ups.append(f"{n_partial_fill} rows have exactly one of "
                               f"active_threshold_uM/inactive_threshold_uM filled -- both must be set "
                               f"together. See sample table.")
        if n_reversed:
            follow_ups.append(f"{n_reversed} rows have active_threshold_uM > inactive_threshold_uM "
                               f"(breakpoints reversed). See sample table.")
        if total_level_mismatches:
            follow_ups.append(f"{total_level_mismatches} rows in mic_activity_labels.csv have a "
                               f"threshold_match_level inconsistent with an independent species-then"
                               f"-genus lookup against organism_thresholds.csv -- see sample table.")
        if total_label_mismatches:
            follow_ups.append(f"{total_label_mismatches} rows in mic_activity_labels.csv have a "
                               f"label inconsistent with independently re-derived interval-vs-breakpoint "
                               f"logic -- see sample table. This is a labeling pipeline bug, not a "
                               f"stale-table issue.")
    elif coverage_pct < 100 and missing_species.shape[0] > 0:
        status = "FLAGGED"
        follow_ups.append(f"Threshold coverage is {coverage_pct:.1f}% -- "
                           f"{missing_species.shape[0]} species still have no breakpoint filled in. "
                           f"See punch list for the highest-impact organisms to prioritize.")

    narrative.append(f"**Table structure:** {fmt_int(len(tt))} rows "
                      f"({fmt_int(len(species_thresh))} species + {fmt_int(len(genus_thresh))} genus "
                      f"breakpoints filled in), {n_dup_keys} duplicate keys, {n_bad_level} bad levels, "
                      f"{n_bad_threshold} unparseable thresholds, {n_partial_fill} partial fills, "
                      f"{n_reversed} reversed breakpoints.")
    narrative.append(f"**Threshold coverage:** {fmt_int(n_covered)}/{fmt_int(n_total)} "
                      f"({coverage_pct:.1f}%) of dataset rows resolve to a breakpoint "
                      f"(breakdown: {match_level_counts}).")
    narrative.append(f"**Label distribution:** {label_counts}.")
    narrative.append(f"**Independent re-derivation cross-check:** {total_level_mismatches} "
                      f"threshold_match_level mismatches, {total_label_mismatches} label mismatches "
                      f"out of {fmt_int(n_total)} rows.")
    narrative.append(f"**Censor-direction recovery:** {fmt_int(n_censored_recovered)}/"
                      f"{fmt_int(n_censored_final)} mic_type='censored' rows in the final dataset "
                      f"successfully recovered bounds ({fmt_int(n_censored_rows_in_cd)} rows in "
                      f"mic_censor_direction.csv total).")

    tables = []
    if dup_examples:
        tables.append(("Duplicate threshold table keys (sample, up to 20)", dup_examples))
    if partial_fill_examples:
        tables.append(("Partial-fill rows (sample, up to 20)", partial_fill_examples))
    if reversed_examples:
        tables.append(("Reversed breakpoint rows (sample, up to 20)", reversed_examples))
    if match_level_mismatches:
        tables.append(("threshold_match_level mismatches (sample, up to 20)", match_level_mismatches))
    if label_mismatches:
        tables.append(("label mismatches (sample, up to 20)", label_mismatches))
    if punch_list:
        tables.append((f"Highest-impact species still missing a threshold (top {TOP_N_MISSING})",
                        punch_list))

    return SectionResult(
        section_id="7", title="Threshold coverage & labeling correctness", status=status,
        metrics={
            "n_threshold_rows": len(tt), "n_species_filled": len(species_thresh),
            "n_genus_filled": len(genus_thresh), "n_dup_keys": n_dup_keys,
            "n_bad_level": n_bad_level, "n_bad_threshold": n_bad_threshold,
            "n_partial_fill": n_partial_fill, "n_reversed": n_reversed,
            "n_covered": n_covered, "n_total": n_total, "coverage_pct": round(coverage_pct, 1),
            "n_threshold_match_level_mismatches": total_level_mismatches,
            "n_label_mismatches": total_label_mismatches,
            "n_censored_recovered": n_censored_recovered, "n_censored_final": n_censored_final,
        },
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
