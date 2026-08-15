"""
Dataset step 1: assemble the classification-ready dataset.

Joins data/final_mic_regression_dataset.csv (sequence/smiles/organism/
ncbi_taxon_id_if_available/mic_value_uM/mic_type) with
data/mic_activity_labels.csv (label) by (peptide_id, organism), filters to
label in config's label_filter.included_labels (active/inactive only by
default -- excludes uncertain/unlabeled), and attaches a train/test split
column per peptide_id from data/split_indices.json.

Filtering to active/inactive alone scopes the output to exactly the
organisms with thresholds filled in config/thresholds/organism_thresholds.csv
-- no separate organism allowlist needed, single source of truth stays the
thresholds CSV. As more organisms get thresholds filled in, this stage
picks them up automatically with zero code change.

Output: data/mic_classification_dataset.csv (dataset:model_ready) --
peptide_id, sequence, smiles, organism, ncbi_taxon_id_if_available,
mic_value_uM, mic_type, label, split.
"""
import argparse
import csv
import json

from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.data.assembly import FIELDNAMES, build_classification_dataset
from soamp.data.config import DatasetConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, DatasetConfig)
REGRESSION_CSV = CFG.paths.data_dir / CFG.input_files.regression_dataset_filename
LABELS_CSV = CFG.paths.data_dir / CFG.input_files.activity_labels_filename
SPLIT_JSON = CFG.paths.data_dir / CFG.input_files.split_indices_filename
OUT_CSV = CFG.paths.data_dir / CFG.output.filename
LOG_PATH = CFG.paths.reports_dir / "data_step1_build_classification_dataset_log.txt"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)


def main() -> None:
    log = configure_logging("data.01_build_classification_dataset")
    TRACKER.log_config(CFG.model_dump(mode="json"))

    with open(REGRESSION_CSV, newline="") as f:
        regression_rows = list(csv.DictReader(f))
    with open(LABELS_CSV, newline="") as f:
        label_rows = list(csv.DictReader(f))
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    log.info(f"Loaded {len(regression_rows)} regression rows, "
              f"{len(label_rows)} label rows, "
              f"{len(split['train_peptide_ids'])} train / "
              f"{len(split['test_peptide_ids'])} test peptide_ids")

    out_rows = build_classification_dataset(
        regression_rows, label_rows,
        split["train_peptide_ids"], split["test_peptide_ids"],
        CFG.label_filter.included_labels,
    )

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    organisms = sorted({r["organism"] for r in out_rows})
    label_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for r in out_rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    TRACKER.log_artifact(
        name="dataset_model_ready", artifact_type="dataset",
        paths=[OUT_CSV],
        metadata={"n_rows": len(out_rows), "organisms": organisms,
                  "label_counts": label_counts, "split_counts": split_counts},
        depends_on=["dataset_validated", "labels_threshold_derived", "splits_v1"],
    )

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Data step 1: classification dataset assembly ===",
        f"Included labels: {CFG.label_filter.included_labels}",
        f"Output rows: {len(out_rows)}",
        f"Organisms in scope: {organisms}",
        f"Label distribution: {label_counts}",
        f"Split distribution: {split_counts}",
        f"Wrote {OUT_CSV}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))
    TRACKER.close()


if __name__ == "__main__":
    main()
