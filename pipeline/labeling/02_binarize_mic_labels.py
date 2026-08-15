"""
Labeling step 2: binarize MIC into active/inactive labels per organism.

Joins data/final_mic_regression_dataset.csv against:
  - config/thresholds/organism_thresholds.csv (via src/soamp/common/thresholds.py,
    the shared threshold decider also meant for the future Stage 4 eval)
  - data/mic_censor_direction.csv (recovered bounds for mic_type='censored'
    rows, from pipeline/labeling/01_recover_censor_direction.py)

and applies src/soamp/labeling/labels.py::derive_label row-wise.

Output: data/mic_activity_labels.csv (labels:threshold_derived) --
peptide_id, organism, mic_value_uM, mic_type, active_threshold_uM,
inactive_threshold_uM, threshold_match_level, label, label_reason.

Threshold values are hand-curated and start blank (see
config/thresholds/README.md), so an early run of this script is expected to
label ~100% of rows 'unlabeled' -- that's correct, not a bug. Coverage
improves as config/thresholds/organism_thresholds.csv gets filled in.
"""
import argparse
import csv
from collections import Counter

from soamp.common.thresholds import load_threshold_table, lookup_threshold
from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.labeling.config import LabelingConfig
from soamp.labeling.labels import derive_label
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/labeling/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, LabelingConfig)
FINAL_CSV = CFG.paths.data_dir / "final_mic_regression_dataset.csv"
CENSOR_CSV = CFG.paths.data_dir / "mic_censor_direction.csv"
TABLE_CSV = CFG.paths.thresholds_dir / CFG.threshold_table.filename
OUT_CSV = CFG.paths.data_dir / "mic_activity_labels.csv"
LOG_PATH = CFG.paths.reports_dir / "labeling_step2_binarize_log.txt"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)

FIELDNAMES = ["peptide_id", "organism", "mic_value_uM", "mic_type",
              "active_threshold_uM", "inactive_threshold_uM",
              "threshold_match_level", "label", "label_reason"]


def main() -> None:
    log = configure_logging("labeling.02_binarize_mic_labels")
    TRACKER.log_config(CFG.model_dump(mode="json"))

    with open(FINAL_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    censor_bounds: dict[tuple[int, str], tuple[float, float]] = {}
    with open(CENSOR_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["peptide_id"]), row["organism"])
            censor_bounds[key] = (float(row["recovered_min_uM"]), float(row["recovered_max_uM"]))

    table = load_threshold_table(TABLE_CSV)
    log.info(f"Loaded threshold table: {len(table.species)} species thresholds, "
              f"{len(table.genus)} genus thresholds filled in")

    thresholds_artifact = TRACKER.log_artifact(
        name="thresholds_organism_specific", artifact_type="config",
        paths=[TABLE_CSV],
        metadata={"n_species_thresholds": len(table.species), "n_genus_thresholds": len(table.genus)},
    )

    out_rows = []
    label_counts: Counter[str] = Counter()
    match_level_counts: Counter[str] = Counter()

    for row in rows:
        pid = int(row["peptide_id"])
        organism = row["organism"].strip()
        mic_value_uM = float(row["mic_value_uM"])
        mic_type = row["mic_type"]

        match = lookup_threshold(table, organism)
        active_threshold_uM = match.active_threshold_uM if match else None
        inactive_threshold_uM = match.inactive_threshold_uM if match else None
        match_level = match.match_level if match else "none"
        match_level_counts[match_level] += 1

        recovered_min_uM = recovered_max_uM = None
        if mic_type == "censored":
            recovered_min_uM, recovered_max_uM = censor_bounds[(pid, organism)]

        label, reason = derive_label(
            mic_value_uM, mic_type, active_threshold_uM, inactive_threshold_uM,
            recovered_min_uM, recovered_max_uM,
        )
        label_counts[label] += 1

        out_rows.append({
            "peptide_id": pid,
            "organism": organism,
            "mic_value_uM": mic_value_uM,
            "mic_type": mic_type,
            "active_threshold_uM": active_threshold_uM if active_threshold_uM is not None else "",
            "inactive_threshold_uM": inactive_threshold_uM if inactive_threshold_uM is not None else "",
            "threshold_match_level": match_level,
            "label": label,
            "label_reason": reason,
        })

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    n_total = len(out_rows)
    n_covered = n_total - match_level_counts["none"]
    coverage_pct = 100 * n_covered / n_total if n_total else 0.0

    TRACKER.log_artifact(
        name="labels_threshold_derived", artifact_type="labels",
        paths=[OUT_CSV],
        metadata={"n_total": n_total, "match_level_counts": dict(match_level_counts),
                  "label_counts": dict(label_counts), "coverage_pct": coverage_pct},
        depends_on=["dataset_validated", thresholds_artifact],
    )

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Labeling step 2: MIC binarization ===",
        f"Total rows: {n_total}",
        f"Threshold match level breakdown: {dict(match_level_counts)}",
        f"Threshold coverage: {n_covered}/{n_total} ({coverage_pct:.1f}%)",
        f"Label distribution: {dict(label_counts)}",
        f"Wrote {OUT_CSV}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))
    TRACKER.close()


if __name__ == "__main__":
    main()
