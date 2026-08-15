"""
Features step 1: compute peptide physicochemical descriptors.

Reads data/mic_classification_dataset.csv, dedups to one row per unique
peptide_id (a peptide repeats once per organism it was tested against),
and computes a fixed-order RDKit descriptor vector per peptide directly
from its SMILES -- no dependence on the `sequence` column, so this works
uniformly across canonical and non-canonical/cyclic peptides.

Output: data/peptide_features.csv (features:peptide) -- peptide_id +
the descriptor columns in config/features/base.yaml's descriptors.names
order.
"""
import argparse
import csv

from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.features.config import FeaturesConfig
from soamp.features.peptide import build_peptide_feature_rows, select_unique_peptides
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
OUT_CSV = CFG.paths.data_dir / CFG.output_files.peptide_features_filename
LOG_PATH = CFG.paths.reports_dir / "features_step1_build_peptide_features_log.txt"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)


def main() -> None:
    log = configure_logging("features.01_build_peptide_features")
    TRACKER.log_config(CFG.model_dump(mode="json"))

    with open(CLASSIFICATION_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    unique_peptides = select_unique_peptides(rows)
    log.info(f"{len(rows)} classification rows -> {len(unique_peptides)} unique peptides")

    feature_rows = build_peptide_feature_rows(unique_peptides, CFG.descriptors.names)

    fieldnames = ["peptide_id", *CFG.descriptors.names]
    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    TRACKER.log_artifact(
        name="features_peptide", artifact_type="features",
        paths=[OUT_CSV],
        metadata={"descriptor_names": CFG.descriptors.names, "n_unique_peptides": len(feature_rows)},
        depends_on=["dataset_model_ready"],
    )

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 1: peptide descriptor computation ===",
        f"Descriptor names: {CFG.descriptors.names}",
        f"Unique peptides: {len(feature_rows)}",
        f"Wrote {OUT_CSV}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))
    TRACKER.close()


if __name__ == "__main__":
    main()
