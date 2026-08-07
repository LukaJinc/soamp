"""
Section 2: SMILES structural integrity.

For every unique (peptide_id, smiles) pair that made it into the final dataset,
independently verify RDKit can parse the SMILES, independently recompute its
molecular weight (rdkit_mw, not units.compute_smiles_weight) and cross-check
against the stored `molecular_weight` column, and check canonical-SMILES
round-trip stability. A SMILES that fails to parse or whose MW disagrees with
the stored value would silently corrupt every downstream unit conversion for
that peptide (Step 5's ug/mL -> uM math divides by this exact number).
"""
import math
from qa_common import QAContext, SectionResult, fmt_int, md_table, rdkit_mw, rdkit_roundtrip_stable

MW_REL_TOL = 1e-3  # stored molecular_weight is rounded to 5 sig figs (precision(mw, 5))


def run(ctx: QAContext) -> SectionResult:
    follow_ups = []
    narrative = []

    unique = ctx.final.drop_duplicates(subset=["peptide_id", "smiles"])[
        ["peptide_id", "smiles", "molecular_weight"]
    ]

    n_total = len(unique)
    unparseable = []
    mw_mismatches = []
    roundtrip_unstable = []
    n_parseable = 0
    n_mw_checked = 0

    for row in unique.itertuples(index=False):
        pid, smiles, stored_mw = row.peptide_id, row.smiles, row.molecular_weight
        parseable, stable, canon = rdkit_roundtrip_stable(smiles)
        if not parseable:
            unparseable.append({"peptide_id": pid, "smiles": smiles[:60] + ("..." if len(smiles) > 60 else "")})
            continue
        n_parseable += 1
        if not stable:
            roundtrip_unstable.append({"peptide_id": pid, "canonical_smiles": canon[:60] + ("..." if len(canon) > 60 else "")})

        recomputed_mw = rdkit_mw(smiles)
        if recomputed_mw is None:
            unparseable.append({"peptide_id": pid, "smiles": smiles[:60] + ("..." if len(smiles) > 60 else "")})
            continue
        if stored_mw is None or (isinstance(stored_mw, float) and math.isnan(stored_mw)):
            continue
        n_mw_checked += 1
        if not math.isclose(recomputed_mw, float(stored_mw), rel_tol=MW_REL_TOL):
            mw_mismatches.append({
                "peptide_id": pid, "stored_molecular_weight": stored_mw,
                "rdkit_mw_recomputed": round(recomputed_mw, 3),
                "rel_diff": round(abs(recomputed_mw - float(stored_mw)) / float(stored_mw), 6),
            })

    status = "PASS"
    if unparseable:
        status = "FAIL"
        follow_ups.append(f"{len(unparseable)} unique SMILES in the final dataset are not "
                           f"RDKit-parseable -- these should not have survived Step 5's MW-computable filter.")
    if mw_mismatches:
        status = "FAIL"
        follow_ups.append(f"{len(mw_mismatches)} unique (peptide_id, smiles) pairs have a stored "
                           f"molecular_weight that disagrees with an independently recomputed RDKit "
                           f"MW by more than {MW_REL_TOL:.1%} relative -- possible bug in how "
                           f"molecular_weight was persisted, or a SMILES mismatch between Step 5 "
                           f"and the final assembly step.")
    if roundtrip_unstable and status == "PASS":
        status = "FLAGGED"
    if roundtrip_unstable:
        follow_ups.append(f"{len(roundtrip_unstable)} SMILES are not canonical-round-trip-stable "
                           f"(RDKit re-parses its own canonical output to a different canonical form) "
                           f"-- usually benign (e.g. unspecified stereocenters) but worth a spot look.")

    narrative.append(f"Checked **{fmt_int(n_total)}** unique (peptide_id, smiles) pairs from "
                      f"`final_mic_regression_dataset.csv`. **{fmt_int(n_parseable)}** parsed "
                      f"successfully in RDKit ({fmt_int(len(unparseable))} did not). Of "
                      f"**{fmt_int(n_mw_checked)}** with a stored `molecular_weight` to check against, "
                      f"**{fmt_int(len(mw_mismatches))}** disagreed with an independently recomputed "
                      f"RDKit MW by more than {MW_REL_TOL:.1%}. **{fmt_int(len(roundtrip_unstable))}** "
                      f"were parseable but not canonical-round-trip-stable.")

    tables = []
    if unparseable:
        tables.append(("Unparseable SMILES (sample, up to 20)", unparseable[:20]))
    if mw_mismatches:
        tables.append(("Molecular weight mismatches (sample, up to 20)", mw_mismatches[:20]))
    if roundtrip_unstable:
        tables.append(("Round-trip-unstable SMILES (sample, up to 10)", roundtrip_unstable[:10]))

    return SectionResult(
        section_id="2", title="SMILES structural integrity", status=status,
        metrics={
            "n_unique_smiles_checked": n_total, "n_parseable": n_parseable,
            "n_unparseable": len(unparseable), "n_mw_checked": n_mw_checked,
            "n_mw_mismatches": len(mw_mismatches), "n_roundtrip_unstable": len(roundtrip_unstable),
        },
        tables=tables, narrative=narrative, follow_ups=follow_ups,
    )
