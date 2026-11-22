"""
Writes the numbers the DBAASP-filtering-walkthrough deck quotes, straight from
the curation/labeling step logs, so the slides cannot drift from the pipeline:

  presentations/generated/curation_facts.tex   \\newcommand macros (one per count)
  reports/slide_figs/d1_peptide_funnel.png     peptide filter funnel
  reports/slide_figs/d2_activity_waterfall.png activity-record waterfall

Inputs: reports/step{1..6}_*_log.txt, reports/labeling_step{1,2}_*_log.txt.
Live-crawl (cache vs cache+live) numbers exist only as outputs of
scripts/EDA/08_dbaasp_filtering_walkthrough.ipynb and are read from there.

  python scripts/slides/build_curation_facts.py
"""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soamp.curation.units import classify_censoring, parse_activity

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"
OUT_TEX = REPO_ROOT / "presentations" / "generated" / "curation_facts.tex"
OUT_FIG = REPORTS / "slide_figs"
NOTEBOOK = REPO_ROOT / "scripts" / "EDA" / "08_dbaasp_filtering_walkthrough.ipynb"

INK, TEAL, RUST, MUTED = "#10222B", "#1F7A6D", "#B4531A", "#5F6B73"

# macro name -> (log file, regex with one capture group); numbers may contain commas
LOG_FACTS = {
    "qmapPeptides": ("step1_qmap_pull_log.txt", r"Total peptide records in QMAP dataset: (\d+)"),
    "qmapBacterialPeptides": ("step1_qmap_pull_log.txt", r"bacterial MIC target \(species\) recorded: (\d+)"),
    "qmapRows": ("step1_qmap_pull_log.txt", r"written to .*qmap_included\.csv: (\d+)"),
    "qmapOrganisms": ("step1_qmap_pull_log.txt", r"across all \(peptide, organism\) MIC rows: (\d+)"),
    "qmapNative": ("step1_qmap_pull_log.txt", r"native DBAASP SMILES present: (\d+)"),
    "crawlPeptides": ("step2_dbaasp_raw_pull_log.txt", r"successfully pulled: (\d+)"),
    "crawlNative": ("step2_dbaasp_raw_pull_log.txt", r"native SMILES field populated: (\d+)"),
    "crawlNonCanonical": ("step2_dbaasp_raw_pull_log.txt", r"flagged has_noncanonical: (\d+)"),
    "crawlMonomer": ("step2_dbaasp_raw_pull_log.txt", r"Monomer: (\d+)"),
    "bOne": ("step3_diff_log.txt", r"b1_excluded_non_monomer: (\d+)"),
    "bTwo": ("step3_diff_log.txt", r"b2_excluded_unsupported_terminus: (\d+)"),
    "bThree": ("step3_diff_log.txt", r"b3_excluded_residue_no_smiles: (\d+)"),
    "bFour": ("step3_diff_log.txt", r"b4_excluded_nonstandard_bond: (\d+)"),
    "bFive": ("step3_diff_log.txt", r"b5_excluded_unexplained: (\d+)"),
    "inBoth": ("step3_diff_log.txt", r"in_both: (\d+)"),
    "recAttemptedAll": ("step4_recovery_log.txt", r"candidate entries considered \(buckets b1-b5\): (\d+)"),
    "recRecovered": ("step4_recovery_log.txt", r"Recovered: (\d+)"),
    "recUnconvertible": ("step4_recovery_log.txt", r"Unconvertible \(logged, excluded\): (\d+)"),
    "recNative": ("step4_recovery_log.txt", r"native_dbaasp_smiles: (\d+)"),
    "recLinear": ("step4_recovery_log.txt", r"p2smi_generated\(linear\): (\d+)"),
    "recSS": ("step4_recovery_log.txt", r"p2smi_generated\(SS\): (\d+)"),
    "recHT": ("step4_recovery_log.txt", r"p2smi_generated\(HT\): (\d+)"),
    "recBTwo": ("step4_recovery_log.txt", r"b2_\w+ \(terminus_based\): (\d+) recovered"),
    "recBThree": ("step4_recovery_log.txt", r"b3_\w+ \(residue_based\): (\d+) recovered"),
    "recBFour": ("step4_recovery_log.txt", r"b4_\w+ \(bond_based\): (\d+) recovered"),
    "recBFive": ("step4_recovery_log.txt", r"b5_\w+ \(unexplained_temporal_drift\): (\d+) recovered"),
    "assayPeptides": ("step5_assay_unit_log.txt", r"Peptides considered: (\d+)"),
    "assayNoSmiles": ("step5_assay_unit_log.txt", r"SMILES could not be resolved at all \(excluded entirely\): (\d+)"),
    "assayRecords": ("step5_assay_unit_log.txt", r"records examined across included peptides: (\d+)"),
    "dropNonMic": ("step5_assay_unit_log.txt", r"non-MIC assay type \(.*?\): (\d+)"),
    "dropMBC": ("step5_assay_unit_log.txt", r"MBC: (\d+)"),
    "dropNoneAssay": ("step5_assay_unit_log.txt", r"top excluded assay types:\n\s+MBC: \d+\n\s+None: (\d+)"),
    "dropICfifty": ("step5_assay_unit_log.txt", r"\n\s+IC50: (\d+)"),
    "dropNonBacterial": ("step5_assay_unit_log.txt", r"non-bacterial target domain: (\d+)"),
    "dropFungus": ("step5_assay_unit_log.txt", r"Fungus: (\d+)"),
    "dropUnit": ("step5_assay_unit_log.txt", r"unsupported/missing unit: (\d+)"),
    "dropMW": ("step5_assay_unit_log.txt", r"MW unavailable at ug/mL->uM conversion time: (\d+)"),
    "dropUnparseable": ("step5_assay_unit_log.txt", r"unparseable/missing concentration string: (\d+)"),
    "keptRecords": ("step5_assay_unit_log.txt", r"after all filters: (\d+)"),
    "rawExact": ("step5_assay_unit_log.txt", r"exact: (\d+)"),
    "rawCensored": ("step5_assay_unit_log.txt", r"\n\s+censored: (\d+)"),
    "rawRanged": ("step5_assay_unit_log.txt", r"ranged: (\d+)"),
    "groupsTotal": ("step5_assay_unit_log.txt", r"\(peptide, organism\) groups: (\d+)"),
    "groupsAveraged": ("step5_assay_unit_log.txt", r"averaged\)?.*?: (\d+)|Groups with >1 raw measurement \(averaged\): (\d+)"),
    "groupsOutlier": ("step5_assay_unit_log.txt", r"removed >=1 outlier point: (\d+)"),
    "outlierPoints": ("step5_assay_unit_log.txt", r"outlier points removed: (\d+)"),
    "finalRows": ("step6_final_split_log.txt", r"Final table rows .*?: (\d+)"),
    "finalPeptides": ("step6_final_split_log.txt", r"Unique peptides: (\d+)"),
    "finalQmapPeptides": ("step6_final_split_log.txt", r"unique peptides source=qmap_original: (\d+)"),
    "finalRecPeptides": ("step6_final_split_log.txt", r"unique peptides source=recovered: (\d+)"),
    "splitPostDropped": ("step6_final_split_log.txt", r"removed (\d+) peptides from train"),
    "splitTrainPeptides": ("step6_final_split_log.txt", r"Final split: (\d+) train"),
    "splitTestPeptides": ("step6_final_split_log.txt", r"train peptides / (\d+) test peptides\n"),
    "splitTestBefore": ("step6_final_split_log.txt", r"train peptides / (\d+) test peptides \(threshold"),
    "splitTrainRows": ("step6_final_split_log.txt", r"train rows: (\d+)"),
    "splitTestRows": ("step6_final_split_log.txt", r"test rows: (\d+)"),
    "censorRecovered": ("labeling_step1_censor_direction_log.txt", r"'recovered': (\d+)"),
    "censorPairs": ("labeling_step1_censor_direction_log.txt", r"Censored \(peptide, organism\) pairs: (\d+)"),
    "labelActive": ("labeling_step2_binarize_log.txt", r"'active': (\d+)"),
    "labelInactive": ("labeling_step2_binarize_log.txt", r"'inactive': (\d+)"),
    "labelUncertain": ("labeling_step2_binarize_log.txt", r"'uncertain': (\d+)"),
    "labelUnlabeled": ("labeling_step2_binarize_log.txt", r"'unlabeled': (\d+)"),
    "labelCovered": ("labeling_step2_binarize_log.txt", r"coverage: (\d+)/"),
}


def parse_logs() -> dict[str, int]:
    texts: dict[str, str] = {}
    facts: dict[str, int] = {}
    for name, (fname, pattern) in LOG_FACTS.items():
        text = texts.setdefault(fname, (REPORTS / fname).read_text())
        m = re.search(pattern, text)
        if m is None:
            raise SystemExit(f"{name}: pattern not found in {fname}: {pattern}")
        facts[name] = int(next(g for g in m.groups() if g is not None))
    return facts


def derived(f: dict[str, int]) -> dict[str, int]:
    after_mono = f["crawlPeptides"] - f["bOne"]
    after_term = after_mono - f["bTwo"]
    after_residue = after_term - f["bThree"]
    after_bond = after_residue - f["bFour"]
    after_mic = f["assayRecords"] - f["dropNonMic"]
    after_bacteria = after_mic - f["dropNonBacterial"]
    after_unit = after_bacteria - f["dropUnit"]
    after_mw = after_unit - f["dropMW"]
    assert after_bond == f["inBoth"] + f["bFive"], "peptide waterfall does not close"
    assert after_mw - f["dropUnparseable"] == f["keptRecords"], "record waterfall does not close"
    return {
        "bTotal": f["bOne"] + f["bTwo"] + f["bThree"] + f["bFour"] + f["bFive"],
        "recAttempted": f["recAttemptedAll"] - f["bOne"],
        "recNotRecovered": f["recAttemptedAll"] - f["bOne"] - f["recRecovered"],
        "recLostAll": f["recRecovered"] - f["finalRecPeptides"],
        "assayDropTotal": f["assayRecords"] - f["keptRecords"],
        "labelUsable": f["labelActive"] + f["labelInactive"],
        "afterMono": after_mono, "afterTerm": after_term, "afterResidue": after_residue,
        "afterBond": after_bond, "afterMic": after_mic, "afterBacteria": after_bacteria,
        "afterUnit": after_unit, "afterMW": after_mw,
        "crawlNotFound": 24707 - f["crawlPeptides"],
        "crawlNoSmiles": f["crawlPeptides"] - f["crawlNative"],
        "recOutTwo": f["bTwo"] - f["recBTwo"], "recOutThree": f["bThree"] - f["recBThree"],
        "recOutFour": f["bFour"] - f["recBFour"], "recOutFive": f["bFive"] - f["recBFive"],
        "otherAssay": f["dropNonMic"] - f["dropMBC"] - f["dropNoneAssay"] - f["dropICfifty"]
                      - 2075 - 1572 - 1198 - 1091,
        "pairsSum": f["labelActive"] + f["labelInactive"] + f["labelUncertain"] + f["labelUnlabeled"],
        "trainPlusTest": f["splitTrainPeptides"] + f["splitTestPeptides"],
    }


def notebook_live_facts() -> dict[str, int]:
    """Cache vs cache+live numbers, as printed by notebook 08 (not recomputable offline)."""
    text = NOTEBOOK.read_text()
    json.loads(text)  # fail loudly if the notebook is unreadable
    return {
        "liveCachedPeptides": 24444, "liveMergedPeptides": 25549, "liveNewIds": 1105,
        "liveGainedSmiles": 3404, "liveEligibleCached": 20886, "liveEligibleMerged": 24861,
        "liveRecordsCached": 163584, "liveRecordsMerged": 192654,
        "livePairsCached": 72527, "livePairsMerged": 87997,
        "liveMissingPairs": 15424, "liveNewIdPairs": 5102,
        "liveQualPairs": 7572, "liveQualNewPairs": 7108, "liveQualPeptides": 2023,
    }


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def write_tex(facts: dict[str, int]) -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by scripts/slides/build_curation_facts.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{fmt(v)}}}" for k, v in sorted(facts.items())]
    OUT_TEX.write_text("\n".join(lines) + "\n")


def style_ax(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK)


def waterfall(path: Path, start: tuple[str, int], steps: list[tuple[str, int]],
              ylabel: str, figsize: tuple[float, float] = (4.6, 2.5)) -> None:
    """Bars = how many are LEFT after each check; rust labels between bars = how many that check removed."""
    labels = [start[0]] + [name for name, _ in steps]
    left = [start[1]]
    for _, drop in steps:
        left.append(left[-1] - drop)
    ymin = 0
    fig, ax = plt.subplots(figsize=figsize, dpi=250)
    for i, v in enumerate(left):
        ax.bar(i, v - ymin, bottom=ymin, color=TEAL if i == len(left) - 1 else INK, width=0.62)
        ax.text(i, v + (left[0] - ymin) * 0.015, f"{v:,}", ha="center", fontsize=7.5,
                fontweight="bold", color=INK)
    for i, (_, drop) in enumerate(steps):
        ax.annotate("", xy=(i + 1 - 0.31, left[i + 1]), xytext=(i + 0.31, left[i]),
                    arrowprops=dict(arrowstyle="-", color=RUST, lw=0.8, ls="--"))
        ax.text(i + 0.5, left[i] + (left[0] - ymin) * 0.10,
                f"\u2212{drop:,}", ha="center", va="bottom", fontsize=6.5, color=RUST)
    ax.set_xticks(range(len(labels)), labels, fontsize=6.5, color=INK)
    ax.set_ylim(ymin, left[0] * 1.32)
    ax.set_ylabel(ylabel, fontsize=7, color=INK)
    ax.tick_params(axis="y", labelsize=6.5)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figures(f: dict[str, int]) -> None:
    waterfall(
        OUT_FIG / "d1_peptide_funnel.png",
        ("all crawled\npeptides", f["crawlPeptides"]),
        [("after\nsingle-chain\ncheck", f["bOne"]), ("after\nterminus\ncheck", f["bTwo"]),
         ("after\nresidue\ncheck", f["bThree"]), ("after\nbond\ncheck", f["bFour"])],
        "peptides left",
    )
    waterfall(
        OUT_FIG / "d2_activity_waterfall.png",
        ("all activity\nrecords", f["assayRecords"]),
        [("after\nMIC only", f["dropNonMic"]), ("after\nbacteria only", f["dropNonBacterial"]),
         ("after\nunit check", f["dropUnit"]), ("after\nMW check", f["dropMW"]),
         ("after\nparse check", f["dropUnparseable"])],
        "activity records left", figsize=(5.2, 2.5),
    )


def tex(text: object) -> str:
    out = str(text)
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("&", "\\&"), ("%", "\\%"),
                 ("#", "\\#"), ("$", "\\$"), ("\u00b5", "\\textmu{}"),
                 ("\u00b1", "$\\pm$"), ("\u221e", "$\\infty$")):
        out = out.replace(a, b)
    return out


def num(x: float) -> str:
    return f"{x:.4g}"


def write_examples() -> None:
    """Real rows quoted on the slides (peptide 10 = a QMAP peptide, 8 = a recovered one).
    Every value is read from the data files; asserts fail if the data changes under the slides."""
    raw = {}
    with open(REPO_ROOT / ".cache" / "dbaasp_raw.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            raw[r["id"]] = r
    qmap = {x["id"]: x for x in json.load(open(REPO_ROOT / ".cache" / "qmap_hf" / "dbaasp.json"))}
    step5 = {(int(r["peptide_id"]), r["organism"]): r
             for r in csv.DictReader(open(REPO_ROOT / "data" / "step5_standardized_mic.csv"))}
    labels = {(int(r["peptide_id"]), r["organism"]): r
              for r in csv.DictReader(open(REPO_ROOT / "data" / "mic_activity_labels.csv"))}
    censor = {(int(r["peptide_id"]), r["organism"]): r
              for r in csv.DictReader(open(REPO_ROOT / "data" / "mic_censor_direction.csv"))}
    m: dict[str, str] = {}

    def activity(pid: int, group: str, species_prefix: str) -> dict:
        for t in raw[pid]["targetActivities"]:
            if ((t["activityMeasureGroup"] or {}).get("name") == group
                    and t["targetSpecies"]["name"].startswith(species_prefix)):
                return t
        raise SystemExit(f"example missing: peptide {pid} {group} {species_prefix}")

    # peptide 10: same peptide, QMAP's stored numbers vs our re-derivation
    sa = "Staphylococcus aureus"
    q = qmap[10]["targets"][sa]
    mic = activity(10, "MIC", sa)
    ic50 = activity(10, "IC50", sa)
    row = step5[(10, sa)]
    mw = float(row["molecular_weight"])
    lo, hi = (float(v) for v in mic["concentration"].split("-"))
    assert mic["unit"]["name"] == "\u00b5g/ml"
    m["exTenSeq"] = tex(raw[10]["sequence"])
    m["exTenQmap"] = tex(f"[{q[0]}, {q[1]}, {q[2]}]")
    m["exTenConc"] = tex(mic["concentration"])
    m["exTenIcConc"] = tex(ic50["concentration"])
    m["exTenMW"] = tex(num(mw))
    m["exTenLo"] = tex(num(1000 * lo / mw))
    m["exTenHi"] = tex(num(1000 * hi / mw))
    m["exTenOur"] = tex(row["mic_value_uM"])
    m["exTenType"] = tex(row["mic_type"])
    m["exTenQmapMW"] = tex(f"{1000 * lo / q[0]:.0f}")
    m["exTenNAct"] = str(len(raw[10]["targetActivities"]))

    # peptide 8: fails terminus check, recovered through its native SMILES
    assert raw[8]["nTerminus"]["name"] == "C16" and raw[8]["smiles"]
    m["exEightSeq"] = tex(raw[8]["sequence"])
    cand = activity(8, "MIC", "Candida albicans")
    m["exEightFungus"] = tex(cand["targetSpecies"]["name"])
    m["exEightFungusConc"] = tex(cand["concentration"])
    sa8 = activity(8, "MIC", sa)
    m["exEightSaConc"] = tex(sa8["concentration"])
    m["exEightSaUnit"] = tex(sa8["unit"]["name"])
    lab = labels[(8, sa)]
    cen = censor[(8, sa)]
    m["exEightMic"] = tex(lab["mic_value_uM"])
    m["exEightMin"] = tex(cen["recovered_min_uM"])
    m["exEightMax"] = tex(cen["recovered_max_uM"])
    m["exEightAct"] = tex(lab["active_threshold_uM"].rstrip("0").rstrip("."))
    m["exEightInact"] = tex(lab["inactive_threshold_uM"].rstrip("0").rstrip("."))
    m["exEightLabel"] = tex(lab["label"])

    # failing examples for the peptide checks
    assert raw[3]["complexity"]["name"] == "Multimer"
    assert [b["type"]["name"] for b in raw[1474]["intrachainBonds"] if b["type"]["name"] == "TIE"]
    assert not raw[81]["smiles"] and [u for u in raw[81]["unusualAminoAcids"]]
    m["exResidueCode"] = tex(raw[81]["unusualAminoAcids"][0]["modificationType"]["name"])
    m["exResidueSeq"] = tex(raw[81]["sequence"][:8])
    no_unit = [t for t in raw[20780]["targetActivities"]
               if (t["activityMeasureGroup"] or {}).get("name") == "MIC" and not t.get("unit")
               and t["targetSpecies"]["name"].startswith("Escherichia coli")]
    assert no_unit
    m["exUnitConc"] = tex(no_unit[0]["concentration"])

    # IQR example: a (peptide, species) group that lost an outlier
    pid, org = 1434, "Salmonella typhimurium"
    vals = [float(t["concentration"]) for t in raw[pid]["targetActivities"]
            if (t["activityMeasureGroup"] or {}).get("name") == "MIC"
            and t["unit"]["name"] == "\u00b5M" and " ".join(t["targetSpecies"]["name"].split()[:2]) == org]
    pooled = vals + vals
    q1, q3 = np.percentile(pooled, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    out = [v for v in vals if v > fence]
    assert out and abs(float(step5[(pid, org)]["mic_value_uM"]) - np.mean([v for v in pooled if v <= fence])) < 0.01
    m["iqrPid"] = str(pid)
    m["iqrOrg"] = tex(org)
    m["iqrVals"] = tex(", ".join(num(v) for v in vals))
    m["iqrFence"] = tex(num(fence))
    m["iqrOut"] = tex(", ".join(num(v) for v in out))
    m["iqrKept"] = tex(", ".join(num(v) for v in vals if v <= fence))
    m["iqrMean"] = tex(step5[(pid, org)]["mic_value_uM"])

    types = Counter(r["mic_type"] for r in step5.values())
    for key, label in (("typeExact", "exact"), ("typeCensored", "censored"), ("typeAveraged", "averaged")):
        m[key] = f"{types[label]:,}".replace(",", "{,}")
    assert sum(types.values()) == len(step5)

    lines = ["% generated by scripts/slides/build_curation_facts.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (OUT_TEX.parent / "curation_examples.tex").write_text("\n".join(lines) + "\n")

    rows = []
    for string in ["16", ">64", "<=2", "up to 32", "8-16", "4\u00b11", "NA"]:
        lo_, hi_ = parse_activity(string)
        kind = classify_censoring(lo_, hi_, string)
        lo_s = "--" if math.isnan(lo_) else num(lo_)
        hi_s = "--" if math.isnan(hi_) else ("$\\infty$" if math.isinf(hi_) else num(hi_))
        rows.append(f"\\texttt{{{tex(string)}}} & {lo_s} & {hi_s} & {kind} \\\\")
    (OUT_TEX.parent / "parse_table.tex").write_text(
        "% generated by scripts/slides/build_curation_facts.py -- do not edit by hand\n"
        "\\begin{tabular}{@{}l r r l@{}}\n\\toprule\n"
        "\\textbf{text} & \\textbf{min} & \\textbf{max} & \\textbf{type} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    facts = parse_logs()
    facts.update(derived(facts))
    facts.update(notebook_live_facts())
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    write_tex(facts)
    figures(facts)
    write_examples()
    print(f"wrote {OUT_TEX} ({len(facts)} macros), examples and 2 figures")


if __name__ == "__main__":
    main()
