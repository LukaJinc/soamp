"""
Labeling step 1: recover censoring direction for mic_type='censored' rows.

Reads data/final_mic_regression_dataset.csv, finds every mic_type='censored'
row, and re-derives its true (min, max) bound in uM by re-walking the raw
DBAASP cache (.cache/dbaasp_raw.jsonl) -- see src/soamp/labeling/censoring.py
for why this is possible and how it's done.

Output: data/mic_censor_direction.csv, one row per censored (peptide_id,
organism) pair -- (peptide_id, organism, recovery_status, censor_type,
recovered_min_uM, recovered_max_uM). Consumed by
pipeline/labeling/02_binarize_mic_labels.py.
"""
import argparse
import csv

from dotenv import load_dotenv

from soamp.labeling.censoring import recover_bounds
from soamp.labeling.config import LabelingConfig
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
RAW_JSONL = CFG.paths.cache_dir / "dbaasp_raw.jsonl"
OUT_CSV = CFG.paths.data_dir / "mic_censor_direction.csv"
LOG_PATH = CFG.paths.reports_dir / "labeling_step1_censor_direction_log.txt"

FIELDNAMES = ["peptide_id", "organism", "recovery_status", "censor_type",
              "recovered_min_uM", "recovered_max_uM"]


def main() -> None:
    log = configure_logging("labeling.01_recover_censor_direction")

    with open(FINAL_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    molecular_weights: dict[int, float | None] = {}
    censored_pairs: set[tuple[int, str]] = set()
    for row in rows:
        pid = int(row["peptide_id"])
        if pid not in molecular_weights:
            mw_raw = row["molecular_weight"].strip()
            molecular_weights[pid] = float(mw_raw) if mw_raw else None
        if row["mic_type"] == "censored":
            censored_pairs.add((pid, row["organism"].strip()))

    log.info(f"Loaded {len(rows)} dataset rows, {len(molecular_weights)} distinct peptides, "
              f"{len(censored_pairs)} censored (peptide, organism) pairs to recover")

    bounds = recover_bounds(RAW_JSONL, censored_pairs, molecular_weights, log=log)

    out_rows = []
    status_counts: dict[str, int] = {}
    for (pid, organism), b in sorted(bounds.items()):
        status_counts[b.status] = status_counts.get(b.status, 0) + 1
        out_rows.append({
            "peptide_id": pid,
            "organism": organism,
            "recovery_status": b.status,
            "censor_type": b.censor_type,
            "recovered_min_uM": b.min_uM,
            "recovered_max_uM": b.max_uM,
        })

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Labeling step 1: censor direction recovery ===",
        f"Censored (peptide, organism) pairs: {len(censored_pairs)}",
        f"Recovery status breakdown: {status_counts}",
        f"Wrote {OUT_CSV}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
