"""
Section 3: independent unit-conversion spot-check.

For step5 rows that are a single, exact, non-averaged measurement (mic_type
== 'exact', n_raw_measurements == 1), trace back to the one underlying raw
DBAASP targetActivity record and independently recompute uM from it --
skipping units.py and parse_activity's string-parsing logic entirely. Only
rows where the raw concentration string is a bare number (is_plain_numeric)
are attempted, since parse_activity's range/censoring parsing is out of
scope for this independent check (disclosed in qa_common.py's docstring).
"""
import math
from qa_common import (
    QAContext, SectionResult, fmt_int, md_table,
    rdkit_mw, ug_ml_to_uM_by_hand, is_plain_numeric, build_raw_activity_index,
)

VALUE_REL_TOL = 1e-2  # stored mic_value_uM is rounded to 3 sig figs (precision(x, 3))


def run(ctx: QAContext) -> SectionResult:
    follow_ups = []
    narrative = []

    candidates = ctx.step5[(ctx.step5["mic_type"] == "exact") & (ctx.step5["n_raw_measurements"] == 1)]
    unique_pids = candidates["peptide_id"].unique().tolist()
    raw_index = build_raw_activity_index(ctx, unique_pids)

    n_candidates = len(candidates)
    n_traced = 0          # found exactly one qualifying, plain-numeric raw record
    n_ambiguous = 0       # 0 or >1 qualifying plain-numeric raw records, or MW unresolvable
    n_mismatches = 0
    mismatches = []
    mw_cache = {}

    for row in candidates.itertuples(index=False):
        pid, organism, smiles = row.peptide_id, row.organism, row.smiles
        stored_val = row.mic_value_uM

        acts = raw_index.get(pid, [])
        qualifying = [a for a in acts
                      if a["assay_group"] == "MIC" and a["domain"] == "Bacteria"
                      and a["binomial"] == organism
                      and a["unit"] in ("µM", "µg/ml")
                      and is_plain_numeric(a["concentration_raw"])]
        if len(qualifying) != 1:
            n_ambiguous += 1
            continue

        act = qualifying[0]
        raw_val = float(act["concentration_raw"])
        if act["unit"] == "µM":
            recomputed = raw_val
        else:
            if smiles not in mw_cache:
                mw_cache[smiles] = rdkit_mw(smiles)
            mw = mw_cache[smiles]
            if mw is None:
                n_ambiguous += 1
                continue
            recomputed = ug_ml_to_uM_by_hand(raw_val, mw)

        n_traced += 1
        if not math.isclose(recomputed, float(stored_val), rel_tol=VALUE_REL_TOL, abs_tol=1e-6):
            n_mismatches += 1
            mismatches.append({
                "peptide_id": pid, "organism": organism, "unit": act["unit"],
                "raw_value": raw_val, "stored_mic_value_uM": stored_val,
                "independently_recomputed_uM": round(recomputed, 4),
            })

    status = "PASS"
    if n_mismatches:
        status = "FAIL"
        follow_ups.append(f"{n_mismatches} of {n_traced} independently traced+recomputed rows "
                           f"disagree with the stored mic_value_uM by more than {VALUE_REL_TOL:.1%} "
                           f"-- see mismatch table.")
    if n_traced == 0:
        status = "FLAGGED"
        follow_ups.append("Could not uniquely trace any candidate row back to a single qualifying "
                           "raw measurement -- spot-check coverage is zero, treat this section as "
                           "inconclusive rather than a pass.")

    narrative.append(f"Of **{fmt_int(n_candidates)}** step5 rows that are a single exact "
                      f"(non-averaged, non-censored) measurement, **{fmt_int(n_traced)}** were "
                      f"uniquely traced to one qualifying raw DBAASP `targetActivity` record with a "
                      f"bare-numeric concentration string (skipping anything requiring "
                      f"`parse_activity`'s range/censoring logic, which is out of scope for this "
                      f"independent check). **{fmt_int(n_ambiguous)}** could not be uniquely traced "
                      f"(0 or >1 qualifying raw records, or unresolvable MW) and were skipped. Of the "
                      f"traced rows, **{fmt_int(n_mismatches)}** disagreed with the pipeline's stored "
                      f"`mic_value_uM` by more than {VALUE_REL_TOL:.1%} when independently recomputed "
                      f"via `rdkit_mw` + hand-rolled `ug_ml_to_uM_by_hand` (bypassing `units.py` "
                      f"entirely).")

    tables = []
    if mismatches:
        tables.append(("Unit-conversion mismatches (sample, up to 20)", mismatches[:20]))

    return SectionResult(
        section_id="3", title="Independent unit-conversion spot-check", status=status,
        metrics={
            "n_candidates": n_candidates, "n_traced": n_traced,
            "n_ambiguous_skipped": n_ambiguous, "n_mismatches": n_mismatches,
        },
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
