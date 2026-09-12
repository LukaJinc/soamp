"""
Features step 1: compute peptide features via the configured featurization
method.

Reads data/mic_classification_dataset.csv, dedups to one row per unique
peptide_id (a peptide repeats once per organism it was tested against), and
builds a fixed-order feature vector per peptide via the method selected in
config/features/base.yaml's peptide_featurization.method -- see
soamp.features.peptide_featurizers for the available strategies
(rdkit_descriptors: RDKit physicochemical descriptors computed directly from
SMILES; peptideclm_embedding: frozen PeptideCLM transformer embeddings).
Both work uniformly across canonical and non-canonical/cyclic peptides, with
no dependence on the `sequence` column.

Output: the data/<output_files.peptide_features_filename> CSV
(features:peptide) -- peptide_id + the active featurizer's feature_names
columns, in order. That filename is method-suffixed by the per-cell config
overlays, so two featurization methods' artifacts coexist rather than
overwriting each other.
"""
import argparse
import csv

from dotenv import load_dotenv

from soamp.features.config import FeaturesConfig
from soamp.features.peptide import select_unique_peptides
from soamp.features.peptide_featurizers import build_peptide_featurizer
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
# Method-suffixed so running this stage for a second featurization method
# doesn't overwrite the first one's log.
LOG_PATH = (
    CFG.paths.reports_dir
    / f"features_step1_build_peptide_features_{CFG.peptide_featurization.method}_log.txt"
)


def main() -> None:
    log = configure_logging("features.01_build_peptide_features")

    with open(CLASSIFICATION_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    unique_peptides = select_unique_peptides(rows)
    log.info(f"{len(rows)} classification rows -> {len(unique_peptides)} unique peptides")

    featurizer = build_peptide_featurizer(
        CFG.peptide_featurization.method, **CFG.peptide_featurization.active_kwargs()
    )
    log.info(f"peptide_featurization.method: {CFG.peptide_featurization.method}")
    featurizer.fit(unique_peptides)
    feature_rows = featurizer.transform(unique_peptides)
    feature_names = featurizer.feature_names

    fieldnames = ["peptide_id", *feature_names]
    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 1: peptide feature computation ===",
        f"Method: {CFG.peptide_featurization.method}",
        f"Feature names: {feature_names}",
        f"Unique peptides: {len(feature_rows)}",
        f"Wrote {OUT_CSV}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
