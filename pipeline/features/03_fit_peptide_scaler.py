"""
Features step 3: fit a StandardScaler on the peptide feature vectors.

Method-agnostic by design: descriptor/feature names are read directly off
data/peptide_features.csv's header (whatever peptide_featurization.method
step 1 used to produce it -- RDKit descriptor names, or PeptideCLM's
dim_0..dim_767), not re-derived from config, so there's exactly one source
of truth for "what columns does this artifact have."

Fits only on peptide_ids appearing in a split=='train' row of
data/mic_classification_dataset.csv -- never on test -- so no test-set
statistics leak into the standardization applied at both train and eval
time. Scaling is applied unconditionally regardless of peptide featurization
method (standardizing a frozen embedding before concatenation with the
organism representation is harmless/standard practice, and it keeps this
script's descriptor-agnostic contract simple).

Output: the data/<output_files.peptide_feature_scaler_filename> JSON
(mean/scale per feature name,
consumed by src/soamp/data/torch_dataset.py::PeptideOrganismDataset).
"""
import argparse
import csv
import json

from dotenv import load_dotenv

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
LOG_PATH = (
    CFG.paths.reports_dir
    / f"features_step3_fit_peptide_scaler_{CFG.peptide_featurization.method}_log.txt"
)


def main() -> None:
    log = configure_logging("features.03_fit_peptide_scaler")

    with open(CLASSIFICATION_CSV, newline="") as f:
        train_peptide_ids = {
            row["peptide_id"] for row in csv.DictReader(f) if row["split"] == "train"
        }

    with open(PEPTIDE_FEATURES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        descriptor_names = [c for c in reader.fieldnames if c != "peptide_id"]
        feature_rows = list(reader)

    train_feature_rows = [
        {name: float(row[name]) for name in descriptor_names}
        for row in feature_rows
        if row["peptide_id"] in train_peptide_ids
    ]
    log.info(f"Fitting scaler on {len(train_feature_rows)} train-split peptides "
              f"(of {len(feature_rows)} total unique peptides), "
              f"{len(descriptor_names)} feature columns")

    scaler = fit_scaler(train_feature_rows, descriptor_names)
    scaler["method"] = CFG.peptide_featurization.method
    scaler["fitted_on"] = "train split of data/mic_classification_dataset.csv"
    scaler["n_peptides_fit"] = len(train_feature_rows)

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(scaler, f, indent=2)
        f.write("\n")

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


if __name__ == "__main__":
    main()
