"""
Section 4: independent MIC/Bacteria filter verification.

This is the "Section 6" hinted at in qa_common.py's docstring: for every
(peptide_id, organism) pair in the final dataset, confirm at least one raw
DBAASP targetActivity record actually satisfies assay_group == 'MIC' and
domain == 'Bacteria' with matching species -- using the same classify_species
classifier the pipeline uses, but checking that the filter was *applied* to
every surviving row, not re-deriving taxonomy from scratch. A row that can't
be traced back to a qualifying raw record would mean either the filter logic
has a bug or a row was fabricated/mis-joined during final assembly.
"""
from qa_common import QAContext, SectionResult, fmt_int, md_table, build_raw_activity_index


def run(ctx: QAContext) -> SectionResult:
    follow_ups = []
    narrative = []

    pairs = ctx.final.drop_duplicates(subset=["peptide_id", "organism"])[["peptide_id", "organism"]]
    unique_pids = pairs["peptide_id"].unique().tolist()
    raw_index = build_raw_activity_index(ctx, unique_pids)

    n_pairs = len(pairs)
    n_untraceable = 0
    untraceable_examples = []

    for row in pairs.itertuples(index=False):
        pid, organism = row.peptide_id, row.organism
        acts = raw_index.get(pid, [])
        qualifying = [a for a in acts
                      if a["assay_group"] == "MIC" and a["domain"] == "Bacteria"
                      and a["binomial"] == organism]
        if not qualifying:
            n_untraceable += 1
            if len(untraceable_examples) < 20:
                untraceable_examples.append({
                    "peptide_id": pid, "organism": organism,
                    "n_raw_activities_for_peptide": len(acts),
                })

    status = "PASS" if n_untraceable == 0 else "FAIL"
    if n_untraceable:
        follow_ups.append(f"{n_untraceable} of {n_pairs} (peptide_id, organism) pairs in the final "
                           f"dataset cannot be traced back to any raw targetActivity record with "
                           f"assay_group=='MIC' and domain=='Bacteria' for that species -- the "
                           f"MIC/Bacteria filter may not have been applied correctly to these rows, "
                           f"or the (peptide_id, organism) grouping key drifted from the raw data "
                           f"between Step 5 and final assembly.")

    narrative.append(f"Checked all **{fmt_int(n_pairs)}** unique (peptide_id, organism) pairs in "
                      f"`final_mic_regression_dataset.csv` against a fresh, independent re-scan of "
                      f"each peptide's raw `targetActivities` (assay_group=='MIC' and "
                      f"domain=='Bacteria', domain from `classify_species` -- the same classifier "
                      f"the pipeline uses, checking the filter's *application*, not taxonomy "
                      f"correctness). **{fmt_int(n_untraceable)}** pairs could not be traced back to "
                      f"any qualifying raw record.")

    tables = []
    if untraceable_examples:
        tables.append(("Untraceable (peptide_id, organism) pairs (sample, up to 20)", untraceable_examples))

    return SectionResult(
        section_id="4", title="Independent MIC/Bacteria filter verification", status=status,
        metrics={"n_pairs_checked": n_pairs, "n_untraceable": n_untraceable},
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
