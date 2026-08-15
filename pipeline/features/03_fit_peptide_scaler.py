"""
Features step 3: fit a StandardScaler on the peptide descriptor vectors.

Fits only on peptide_ids appearing in a split=='train' row of
data/mic_classification_dataset.csv -- never on test -- so no test-set
statistics leak into the standardization applied at both train and eval
time.

Output: data/peptide_feature_scaler.json (mean/scale per descriptor,
consumed by src/soamp/data/torch_dataset.py::PeptideOrganismDataset).
"""
import argparse
import csv
import json

from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.features.config import FeaturesConfig
from soamp.features.scaling import fit_scaler
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/features/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, FeaturesConfig)
CLASSIFICATION_CSV = CFG.paths.data_dir / CFG.input_files.classification_dataset_filename
PEPTIDE_FEATURES_CSV = CFG.paths.data_dir / CFG.output_files.peptide_features_filename
OUT_JSON = CFG.paths.data_dir / CFG.output_files.peptide_feature_scaler_filename
LOG_PATH = CFG.paths.reports_dir / "features_step3_fit_peptide_scaler_log.txt"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)


def main() -> None:
    log = configure_logging("features.03_fit_peptide_scaler")
    TRACKER.log_config(CFG.model_dump(mode="json"))

    with open(CLASSIFICATION_CSV, newline="") as f:
        train_peptide_ids = {
            row["peptide_id"] for row in csv.DictReader(f) if row["split"] == "train"
        }

    with open(PEPTIDE_FEATURES_CSV, newline="") as f:
        feature_rows = list(csv.DictReader(f))

    descriptor_names = CFG.descriptors.names
    train_feature_rows = [
        {name: float(row[name]) for name in descriptor_names}
        for row in feature_rows
        if row["peptide_id"] in train_peptide_ids
    ]
    log.info(f"Fitting scaler on {len(train_feature_rows)} train-split peptides "
              f"(of {len(feature_rows)} total unique peptides)")

    scaler = fit_scaler(train_feature_rows, descriptor_names)
    scaler["fitted_on"] = "train split of data/mic_classification_dataset.csv"
    scaler["n_peptides_fit"] = len(train_feature_rows)

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(scaler, f, indent=2)
        f.write("\n")

    TRACKER.log_artifact(
        name="features_peptide_scaler", artifact_type="features",
        paths=[OUT_JSON],
        metadata={"descriptor_names": descriptor_names, "n_peptides_fit": scaler["n_peptides_fit"]},
        depends_on=["dataset_model_ready", "features_peptide"],
    )

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 3: peptide feature scaler ===",
        f"Descriptor names: {descriptor_names}",
        f"n_peptides_fit: {scaler['n_peptides_fit']}",
        f"mean: {scaler['mean']}",
        f"scale: {scaler['scale']}",
        f"Wrote {OUT_JSON}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))
    TRACKER.close()


if __name__ == "__main__":
    main()
