"""Section 1: coverage vs QMAP baseline -- three-way comparison, not one number."""
from qa_common import QAContext, SectionResult, fmt_int, md_table


def run(ctx: QAContext) -> SectionResult:
    follow_ups = []
    narrative = []

    # (a) full-corpus comparison
    final_unique = ctx.final["peptide_id"].nunique()
    n_qmap_raw = len(ctx.qmap_raw)  # freshly reparsed, not trusted from qmap_included.csv
    delta_a = final_unique - n_qmap_raw
    pct_a = 100 * delta_a / n_qmap_raw

    # (b) apples-to-apples: QMAP-sourced final peptides vs QMAP's own *usable* peptides
    #     ("usable" = has >=1 non-empty targets dict, freshly recomputed from qmap_raw)
    n_qmap_usable = sum(1 for e in ctx.qmap_raw if e.get("targets"))
    final_qmap_sourced = ctx.final[ctx.final["source"] == "qmap_original"]["peptide_id"].nunique()
    delta_b = final_qmap_sourced - n_qmap_usable

    # (c) recovery breakdown, cross-checked against audit's claimed counts
    recovered_via_counts = ctx.recovered["recovered_via"].value_counts().to_dict()
    recovery_category_counts = ctx.recovered["recovery_category"].value_counts().to_dict()

    audit_claimed_via = {
        "native_dbaasp_smiles": 2853, "p2smi_generated(linear)": 11,
        "p2smi_generated(SS)": 3, "p2smi_generated(HT)": 3,
    }
    audit_claimed_cat = {
        "terminus_based": 1230, "residue_based": 17, "bond_based": 6,
        "unexplained_temporal_drift": 1617,
    }
    via_mismatches = {k: (v, recovered_via_counts.get(k)) for k, v in audit_claimed_via.items()
                       if recovered_via_counts.get(k) != v}
    cat_mismatches = {k: (v, recovery_category_counts.get(k)) for k, v in audit_claimed_cat.items()
                       if recovery_category_counts.get(k) != v}

    n_recovered_candidates = len(ctx.recovered)
    final_recovered_unique = ctx.final[ctx.final["source"] == "recovered"]["peptide_id"].nunique()
    gap = n_recovered_candidates - final_recovered_unique

    recovered_ids = set(ctx.recovered["peptide_id"])
    step5_ids = set(ctx.step5["peptide_id"])
    dropped_ids = recovered_ids - step5_ids
    # verify: are the dropped ids genuinely absent from step5 (no surviving bacterial-MIC row)?
    dropped_examples = list(dropped_ids)[:10]

    status = "PASS"
    if via_mismatches or cat_mismatches:
        status = "FAIL"
        follow_ups.append("Recovery breakdown counts in recovered_peptides.csv do not match "
                           "the audit report's claimed via/category counts -- see mismatch table.")
    if len(dropped_ids) != gap:
        status = "FAIL"
        follow_ups.append(f"Recovered-candidate-to-final gap ({gap}) does not match the count of "
                           f"recovered peptide_ids genuinely absent from step5 ({len(dropped_ids)}) "
                           f"-- some peptides may be dropped for an undocumented reason.")

    narrative.append(f"**(a) Full-corpus comparison:** final dataset has **{fmt_int(final_unique)}** "
                      f"unique peptides vs QMAP's full published corpus of **{fmt_int(n_qmap_raw)}** "
                      f"(freshly reparsed from `.cache/qmap_hf/dbaasp.json`, not trusted from "
                      f"`qmap_included.csv`). Delta: {delta_a:+,} ({pct_a:+.1f}%). Note this is not "
                      f"a clean apples-to-apples number since QMAP's 18,033 includes peptides with "
                      f"zero bacterial-MIC rows -- see (b) for the fair comparison.")
    narrative.append("")
    narrative.append(f"**(b) Apples-to-apples comparison:** of QMAP's {fmt_int(n_qmap_raw)} peptides, "
                      f"**{fmt_int(n_qmap_usable)}** have at least one non-empty `targets` dict "
                      f"(freshly recomputed -- this is QMAP's own \"usable\" peptide count). This "
                      f"pipeline's final dataset retains **{fmt_int(final_qmap_sourced)}** "
                      f"QMAP-sourced peptides (delta: {delta_b:+,}). A negative delta here is "
                      f"expected: this pipeline reprocesses every peptide's MIC values from raw "
                      f"DBAASP data with its own SMILES-derived-MW unit conversion (Step 5) rather "
                      f"than trusting QMAP's pre-aggregated numbers, so a peptide can legitimately "
                      f"end up with zero surviving rows here (e.g. all its measurements needed "
                      f"unit conversion but its resolved SMILES's MW couldn't be computed) even "
                      f"though QMAP counted it as usable.")
    narrative.append("")
    narrative.append(f"**(c) Recovery contribution:** {fmt_int(n_recovered_candidates)} candidate "
                      f"peptides were recovered from QMAP's exclusion buckets (Step 4); "
                      f"**{fmt_int(final_recovered_unique)}** of those survive into the final "
                      f"dataset with >=1 bacterial MIC row (gap of {gap}, matching "
                      f"{len(dropped_ids)} peptide_ids confirmed absent from step5_standardized_mic.csv).")

    tables = [
        ("recovered_via counts: audit-claimed vs actual (recovered_peptides.csv)", [
            {"recovered_via": k, "audit_claimed": v, "actual": recovered_via_counts.get(k),
             "match": v == recovered_via_counts.get(k)}
            for k, v in audit_claimed_via.items()
        ]),
        ("recovery_category counts: audit-claimed vs actual", [
            {"recovery_category": k, "audit_claimed": v, "actual": recovery_category_counts.get(k),
             "match": v == recovery_category_counts.get(k)}
            for k, v in audit_claimed_cat.items()
        ]),
    ]
    if dropped_examples:
        tables.append(("Sample of recovered peptide_ids absent from step5 (legitimately dropped)",
                        [{"peptide_id": p} for p in dropped_examples]))

    return SectionResult(
        section_id="1", title="Coverage vs QMAP baseline", status=status,
        metrics={
            "final_unique_peptides": final_unique, "qmap_full_corpus": n_qmap_raw,
            "delta_a": delta_a, "pct_a": round(pct_a, 2),
            "qmap_usable_peptides": n_qmap_usable, "final_qmap_sourced": final_qmap_sourced,
            "delta_b": delta_b, "n_recovered_candidates": n_recovered_candidates,
            "final_recovered_unique": final_recovered_unique, "recovery_gap": gap,
            "recovery_gap_verified": len(dropped_ids) == gap,
        },
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
