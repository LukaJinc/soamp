"""
Features step 2: build the organism featurization via the configured method.

Reads data/mic_classification_dataset.csv's unique organism strings and fits
the featurizer selected in config/features/base.yaml's
organism_featurization.method -- see soamp.features.organism_featurizers.
Two methods ship: "vocab_embedding" (each organism gets a stable integer
index for the model's nn.Embedding lookup, reserving
organism_featurization.vocab_embedding.unknown_index -- default 0 -- for
organisms not seen at build time, so an organism added later degrades to
the OOV embedding at inference instead of crashing) and "kmer_composition"
(a genome k-mer composition vector per organism, modeled on LLAMP -- see
soamp.features.organism_featurizers.KmerOrganismFeaturizer; requires
pipeline/features/00_fetch_organism_genomes.py to have already populated
the genome FASTA cache). This script is fully method-agnostic: it writes
whatever featurizer.to_artifact_dict() returns, self-described by a
"method" field -- no per-method branching here.

Output: the data/<output_files.organism_vocab_filename> JSON
(features:organism), method-suffixed by the per-cell config overlays.
"""
import argparse
import csv
import json

from dotenv import load_dotenv

from soamp.features.config import FeaturesConfig
from soamp.features.organism_featurizers import build_organism_featurizer
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
OUT_JSON = CFG.paths.data_dir / CFG.output_files.organism_vocab_filename
LOG_PATH = (
    CFG.paths.reports_dir
    / f"features_step2_build_organism_vocab_{CFG.organism_featurization.method}_log.txt"
)


def main() -> None:
    log = configure_logging("features.02_build_organism_vocab")

    with open(CLASSIFICATION_CSV, newline="") as f:
        organisms = [row["organism"] for row in csv.DictReader(f)]

    featurizer = build_organism_featurizer(
        CFG.organism_featurization.method, **CFG.organism_featurization.active_kwargs()
    )
    featurizer.fit(organisms)
    output = featurizer.to_artifact_dict()

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 2: organism vocab ===",
        f"Method: {CFG.organism_featurization.method}",
        f"Unique organisms in dataset: {len(set(organisms))}",
        f"Wrote {OUT_JSON}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
