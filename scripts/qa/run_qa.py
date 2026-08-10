"""
Driver for the independent QA/validation pass over the curated AMP MIC
regression dataset. Loads QAContext once, runs every section module in
order, assembles one markdown report, and writes it to reports/qa_report.md.

A section raising an exception does not abort the run -- it's recorded as a
FAIL section with the traceback in follow_ups, so one broken check doesn't
hide the results of the others.
"""
import os
import sys
import time
import traceback
import importlib

sys.path.insert(0, os.path.dirname(__file__))
from qa_common import load_context, SectionResult, md_table, REPORTS

SECTION_MODULES = [
    "section1_coverage",
    "section2_smiles_integrity",
    "section3_unit_conversion",
    "section4_filter_verification",
    "section5_split_integrity",
    "section6_sanity_checks",
]

OUT_PATH = os.path.join(REPORTS, "qa_report.md")


def run_section(module_name, ctx):
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        result = mod.run(ctx)
    except Exception:
        tb = traceback.format_exc()
        result = SectionResult(
            section_id=module_name, title=module_name, status="FAIL",
            follow_ups=[f"Section raised an exception and could not complete:\n```\n{tb}\n```"],
        )
    elapsed = time.time() - t0
    print(f"  [{result.status}] {module_name} ({elapsed:.1f}s)", flush=True)
    return result


def render_report(results: list[SectionResult]) -> str:
    lines = []
    lines.append("# QA report: SMILES-based AMP MIC regression dataset")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("Independent audit of `scripts/01-08_*.py`'s outputs (`data/*.csv`, "
                  "`reports/curation_audit.md`). Deliberately avoids importing the pipeline's "
                  "own filter/unit-conversion logic (`scripts/common/units.py`, "
                  "`scripts/06_assay_filter_units.py`) so a bug there would actually be caught -- "
                  "see `scripts/qa/qa_common.py` for the two disclosed exceptions.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | Section | Status |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r.section_id} | {r.title} | **{r.status}** |")
    lines.append("")

    overall_fail = any(r.status == "FAIL" for r in results)
    overall_flag = any(r.status == "FLAGGED" for r in results)
    overall = "FAIL" if overall_fail else ("FLAGGED" if overall_flag else "PASS")
    lines.append(f"**Overall: {overall}**")
    lines.append("")

    for r in results:
        lines.append(f"## Section {r.section_id}: {r.title} -- {r.status}")
        lines.append("")
        if r.narrative:
            lines.extend(r.narrative)
            lines.append("")
        if r.metrics:
            lines.append("**Metrics:**")
            lines.append("")
            lines.append(md_table([{"metric": k, "value": v} for k, v in r.metrics.items()]))
        for caption, rows in r.tables:
            lines.append(f"**{caption}:**")
            lines.append("")
            lines.append(md_table(rows))
        if r.follow_ups:
            lines.append("**Follow-ups:**")
            lines.append("")
            for f in r.follow_ups:
                lines.append(f"- {f}")
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    print("Loading QA context ...", flush=True)
    ctx = load_context()

    print("Running sections ...", flush=True)
    results = [run_section(m, ctx) for m in SECTION_MODULES]

    report = render_report(results)
    os.makedirs(REPORTS, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to {OUT_PATH}", flush=True)

    print("\n=== Summary ===")
    for r in results:
        print(f"  [{r.status}] Section {r.section_id}: {r.title}")


if __name__ == "__main__":
    main()
