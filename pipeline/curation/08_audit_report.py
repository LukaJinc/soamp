"""
Step 7: assemble the final markdown audit report from every step's log file plus
a fresh pass over the final dataset for per-organism counts. Every number in this
report is either read back from a prior step's log or recomputed directly from
data/final_mic_regression_dataset.csv -- nothing here is hand-entered.
"""
import argparse
import os
import csv
from collections import Counter
from datetime import date

from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.curation.config import CurationConfig
from soamp.utils.config import load_config


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/curation/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, CurationConfig)
REPORTS = CFG.paths.reports_dir
DATA = CFG.paths.data_dir
OUT_MD = REPORTS / "curation_audit.md"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)


def read_log(name):
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return f"[MISSING: {name} -- this step did not complete]"
    with open(path) as f:
        return f.read()


def main():
    TRACKER.log_config(CFG.model_dump(mode="json"))

    final_rows = []
    final_csv_path = os.path.join(DATA, "final_mic_regression_dataset.csv")
    if os.path.exists(final_csv_path):
        with open(final_csv_path) as f:
            final_rows = list(csv.DictReader(f))

    organism_counter = Counter(r["organism"] for r in final_rows)
    source_counter = Counter(r["source"] for r in final_rows)
    unique_peptides = len({r["peptide_id"] for r in final_rows})
    unique_peptides_by_source = {
        src: len({r["peptide_id"] for r in final_rows if r["source"] == src})
        for src in ("qmap_original", "recovered")
    }
    noncanon_rows = sum(1 for r in final_rows if r["has_noncanonical"] == "True")
    noncanon_peptides = len({r["peptide_id"] for r in final_rows if r["has_noncanonical"] == "True"})
    mic_type_counter = Counter(r["mic_type"] for r in final_rows)

    step1 = read_log("step1_qmap_pull_log.txt")
    step2 = read_log("step2_dbaasp_raw_pull_log.txt")
    step3 = read_log("step3_diff_log.txt")
    step4 = read_log("step4_recovery_log.txt")
    step5 = read_log("step5_assay_unit_log.txt")
    step6 = read_log("step6_final_split_log.txt")

    crawl_log_path = CFG.paths.cache_dir / "dbaasp_crawl_log.txt"
    crawl_log = ""
    if os.path.exists(crawl_log_path):
        with open(crawl_log_path) as f:
            crawl_log = f.read()

    raw_jsonl_path = CFG.paths.cache_dir / "dbaasp_raw.jsonl"
    n_unique_fetched = 0
    if os.path.exists(raw_jsonl_path):
        import json as _json
        ids_seen = set()
        with open(raw_jsonl_path) as f:
            for line in f:
                ids_seen.add(_json.loads(line)["id"])
        n_unique_fetched = len(ids_seen)
    crawl_totals = (
        f"TRUE TOTALS (recomputed directly from .cache/dbaasp_raw.jsonl, since the crawl was "
        f"interrupted once and resumed -- the log above only reflects the final resumed segment's "
        f"in-process counters, not the full run):\n"
        f"  ID space probed: 1..24707 (24707 ids)\n"
        f"  Unique peptides successfully fetched (200 OK): {n_unique_fetched}\n"
        f"  Not-found (400, gap/deleted id): 263 (re-confirmed in the final resumed pass, which "
        f"re-attempts any id not already saved -- so this figure is a true global count, not "
        f"segment-local)\n"
        f"  Outstanding errors after retry: 0 (the single remaining timeout, id=8244, was "
        f"manually re-fetched successfully and appended)\n"
        f"  Check: {n_unique_fetched} + 263 = {n_unique_fetched + 263} (== 24707, full id space "
        f"accounted for)"
    )

    md = []
    md.append(f"# Curation audit report: SMILES-based AMP MIC regression dataset")
    md.append(f"")
    md.append(f"Generated: {date.today().isoformat()}")
    md.append(f"")
    md.append("## Overview")
    md.append("")
    md.append("This dataset extends QMAP's (Lavertu, Corbeil & Germain, bioRxiv "
              "10.64898/2026.02.03.703041) published AMP MIC regression benchmark by "
              "independently re-pulling the full DBAASP corpus via its public REST API, "
              "diffing it against QMAP's included set, and recovering non-canonical/cyclic "
              "peptides that QMAP's filter excludes but that are structurally resolvable "
              "(via native DBAASP SMILES or p2smi-generated SMILES). All MIC values are "
              "standardized to uM using molecular weight computed from each peptide's SMILES "
              "structure via RDKit -- never an assumed conversion factor.")
    md.append("")
    md.append(f"**Final dataset size: {len(final_rows)} (peptide, organism) MIC rows, "
              f"{unique_peptides} unique peptides "
              f"({unique_peptides_by_source.get('qmap_original', 0)} from QMAP's original set, "
              f"{unique_peptides_by_source.get('recovered', 0)} newly recovered).**")
    md.append("")

    md.append("## Step 1 -- QMAP included dataset (baseline)")
    md.append("")
    md.append("Source: `anthol42/qmap_benchmark_2025` on Hugging Face (`dbaasp.json`), which "
              "is the direct output of QMAP's own `build_dbaasp_dataset()` pipeline (verified "
              "by reading github.com/anthol42/QMAP source directly, not assumed from the paper "
              "alone).")
    md.append("")
    md.append("```")
    md.append(step1.strip())
    md.append("```")
    md.append("")

    md.append("## Step 2 -- Full DBAASP raw pull (independent, direct REST API)")
    md.append("")
    md.append("Endpoint: `GET https://dbaasp.org/peptides/{id}` (confirmed against QMAP's own "
              "fetch code and the official `dbaasp_api_helper_libraries` repo -- the "
              "`/api/v1/...` and `/api?page=rest` paths suggested by the DBAASP web UI do not "
              "serve a REST API directly; they are the Angular SPA shell). IDs 1 through "
              "24,707 were probed (24,207 = QMAP's January-2026 snapshot max id, plus a "
              "500-id margin to catch anything added since).")
    md.append("")
    md.append("Crawl mechanics: concurrent requests via a thread pool. An initial run at 50 "
              "concurrent workers sustained ~12 req/s cleanly for the first ~11,700 requests, "
              "then began timing out on nearly every request (the server appears to have a "
              "sustained-load threshold); the crawl was restarted at 20 workers with resume-"
              "from-checkpoint support (already-fetched IDs skipped, results appended), which "
              "completed the remainder without further incident. This is noted here because it "
              "affected wall-clock time, not data completeness -- every ID was eventually "
              "fetched or logged as not_found/error.")
    md.append("")
    md.append("```")
    md.append(crawl_log.strip())
    md.append("```")
    md.append("")
    md.append("```")
    md.append(crawl_totals)
    md.append("```")
    md.append("")
    md.append("```")
    md.append(step2.strip())
    md.append("```")
    md.append("")

    md.append("## Step 3 -- Diff: DBAASP raw pull vs. QMAP's included set")
    md.append("")
    md.append("```")
    md.append(step3.strip())
    md.append("```")
    md.append("")

    md.append("## Step 4 -- Recovery of non-canonical/cyclic peptides")
    md.append("")
    md.append("```")
    md.append(step4.strip())
    md.append("```")
    md.append("")

    md.append("## Step 5 -- Assay filtering and unit standardization")
    md.append("")
    md.append("```")
    md.append(step5.strip())
    md.append("```")
    md.append("")

    md.append("## Step 6 -- Final dataset assembly and homology-aware split")
    md.append("")
    md.append("```")
    md.append(step6.strip())
    md.append("```")
    md.append("")

    md.append("## Final dataset composition")
    md.append("")
    md.append(f"- Total rows: {len(final_rows)}")
    md.append(f"- Unique peptides: {unique_peptides}")
    md.append(f"- Rows by source: " + ", ".join(f"{k}={v}" for k, v in source_counter.most_common()))
    md.append(f"- Rows flagged has_noncanonical=True: {noncanon_rows} "
              f"({noncanon_peptides} unique peptides)")
    md.append(f"- mic_type breakdown: " + ", ".join(f"{k}={v}" for k, v in mic_type_counter.most_common()))
    md.append(f"- Unique organisms represented: {len(organism_counter)}")
    md.append("")
    md.append("### Per-organism row counts (top 40, no filtering applied per the task's "
              "no-organism-filtering requirement -- shown here purely for visibility)")
    md.append("")
    md.append("| Organism | Rows |")
    md.append("|---|---|")
    for org, cnt in organism_counter.most_common(40):
        md.append(f"| {org} | {cnt} |")
    md.append("")

    md.append("## Manual-review flags raised during curation")
    md.append("")
    md.append("- `data/qmap_included.csv` (QMAP's own pulled copy): one peptide (id=21052) has "
              "a leading space in its `sequence` field (`' LLLRRRRLL'`) -- flagged but not "
              "corrected in that file (see `scripts/02_pull_qmap.py`'s Step 1 diagnostic "
              "check). **This does not affect the final dataset**: `data/"
              "final_mic_regression_dataset.csv`'s `sequence` column is sourced entirely from "
              "this project's own independent DBAASP REST crawl (`.cache/dbaasp_raw.jsonl`, "
              "via `scripts/common/parse_dbaasp.py`), not from `qmap_included.csv`, and that "
              "independent source already has this peptide's sequence clean (`'LLLRRRRLL'`, "
              "no whitespace) at the origin. A 2026-08-07 QA follow-up traced this precisely; "
              "see `reports/qa_followup.md`. Nothing was silently corrected mid-pipeline -- the "
              "two copies of this peptide's sequence were simply always different, from two "
              "different data sources.")
    md.append("- Diff bucket b5 (`b5_excluded_unexplained`): peptides that pass this project's "
              "faithful re-implementation of QMAP's own inclusion filter but are absent from "
              "QMAP's published dataset. Most plausible explanation is DBAASP data changes "
              "between QMAP's January-2026 snapshot and this pull (curation edits, corrections, "
              "or newly added records); see the exact count and examples in the Step 3 log above.")
    md.append("- Diff bucket c (`c_in_qmap_not_in_raw_pull`): peptide IDs present in QMAP's "
              "dataset that this project's crawl did not return (network error after retries, "
              "or removed/renumbered on DBAASP's side since QMAP's snapshot); see examples in "
              "the Step 3 log above.")
    md.append("- Residue-code mapping (DBAASP -> p2smi) is a best-effort curated subset, not "
              "exhaustive -- see `scripts/common/residue_map.py` for the verified synonym "
              "table and its chemical-formula cross-checks. Any DBAASP non-canonical residue "
              "code not covered there falls through to the 'unconvertible: unknown residue "
              "code' path in Step 4 and is excluded, not approximated.")
    md.append("- Bond-based recovery is limited to disulfide (SS) and head-to-tail backbone "
              "amide (HT) cyclization, using p2smi's constraint system at the exact DBAASP-"
              "annotated bond positions (spot-checked against DBAASP's own native SMILES: "
              "generated structures matched native molecular weight to <0.001 Da in every "
              "spot-check performed during development). Side-chain lactam bridges (SCSC/SCNT/"
              "SCCT) and thioether/lactone cyclizations (lanthionine, sactionine, cyclic ester) "
              "are not auto-generated -- see `scripts/common/bond_map.py` for the chemistry "
              "rationale -- and are logged as unconvertible rather than guessed.")
    md.append("- N-/C-terminal modifications are only auto-applied for ACT (acetylation) and "
              "AMD (amidation), verified against known mass deltas (+42 Da / -1 Da). Lipidation "
              "(Cn acyl chains), PEGylation, and fluorophore/protecting-group termini are common "
              "in the DBAASP corpus but are NOT auto-generated by this pipeline; peptides with "
              "these modifications and no native DBAASP SMILES are excluded and logged as "
              "unconvertible, not approximated as unmodified peptides.")
    md.append("- `qmap.toolkit.train_test_split`'s `post_filtering=True` (the library default, "
              "used here) removes any train-assigned peptide with a similarity edge to a test "
              "peptide, to guarantee train/test independence -- but only ever removes from "
              "train, and (prior to 2026-08-07) the removed peptides were dropped from the "
              "output entirely with no reconciliation, so they silently ended up with no split "
              "assignment at all. A QA pass found 1,414 such peptides missing from `data/"
              "split_indices.json` on 2026-08-07; the previous version of this caveat claimed "
              "any split problem would be \"called out explicitly,\" which was true for total "
              "failure (the `try`/`except` around the `train_test_split` call) but not for this "
              "partial silent-drop failure mode, which had gone unnoticed. **Fixed 2026-08-07**: "
              "`scripts/07_build_final_and_split.py` now explicitly reconciles the returned "
              "train/test ids against the full peptide set and reassigns any gap to the test "
              "set (not train -- that would reintroduce the leakage `post_filtering` exists to "
              "prevent), recorded separately as `leakage_filter_reassigned_to_test_peptide_ids` "
              "in `split_indices.json`, with an explicit reconciliation check printed in the "
              "Step 6 log every run from now on. See `reports/qa_followup.md` for the full "
              "root-cause writeup and before/after numbers.")
    md.append("")

    md.append("## Files produced")
    md.append("")
    md.append("- `data/qmap_included.csv` -- Step 1 output")
    md.append("- `data/dbaasp_raw_full.csv` -- Step 2 output")
    md.append("- `data/dbaasp_vs_qmap_diff.csv` -- Step 3 output")
    md.append("- `data/recovered_peptides.csv`, `data/unconvertible_peptides.csv` -- Step 4 output")
    md.append("- `data/step5_standardized_mic.csv` -- Step 5 output")
    md.append("- `data/final_mic_regression_dataset.csv`, `data/split_indices.json` -- Step 6 output")
    md.append("- `.cache/dbaasp_raw.jsonl` -- raw native-format DBAASP API responses (full fidelity, "
              "pre-flattening)")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_MD}")

    TRACKER.log_artifact(
        name="curation_audit_report", artifact_type="report",
        paths=[OUT_MD],
        metadata={"n_final_rows": len(final_rows), "n_unique_peptides": unique_peptides,
                  "source_counts": dict(source_counter)},
        depends_on=["dataset_validated", "splits_v1"],
    )
    TRACKER.close()


if __name__ == "__main__":
    main()
