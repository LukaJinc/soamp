"""
Features step 2: build the organism vocabulary for the nn.Embedding lookup.

Reads data/mic_classification_dataset.csv's unique organism strings and
assigns each a stable index, reserving config's organism_vocab.unknown_index
(default 0) for organisms not seen at build time -- lets an organism added
later (once its threshold is filled in) degrade to the OOV embedding at
inference instead of crashing.

Output: data/organism_vocab.json (features:organism).
"""
import argparse
import csv
import json

from dotenv import load_dotenv

from soamp.common.tracking import build_tracker
from soamp.features.config import FeaturesConfig
from soamp.features.organism import build_vocab
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
LOG_PATH = CFG.paths.reports_dir / "features_step2_build_organism_vocab_log.txt"
TRACKER = build_tracker(CFG.tracking, CFG.paths.tracking_dir)


def main() -> None:
    log = configure_logging("features.02_build_organism_vocab")
    TRACKER.log_config(CFG.model_dump(mode="json"))

    with open(CLASSIFICATION_CSV, newline="") as f:
        organisms = [row["organism"] for row in csv.DictReader(f)]

    unknown_index = CFG.organism_vocab.unknown_index
    vocab = build_vocab(organisms, unknown_index=unknown_index)

    output = {
        "unknown_index": unknown_index,
        "vocab_size": len(vocab) + 1,
        "vocab": vocab,
    }
    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    TRACKER.log_artifact(
        name="features_organism", artifact_type="features",
        paths=[OUT_JSON],
        metadata={"vocab_size": output["vocab_size"], "unknown_index": unknown_index},
        depends_on=["dataset_model_ready"],
    )

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Features step 2: organism vocab ===",
        f"Organisms: {sorted(vocab.keys())}",
        f"Vocab size (incl. reserved unknown_index={unknown_index}): {output['vocab_size']}",
        f"Wrote {OUT_JSON}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))
    TRACKER.close()


if __name__ == "__main__":
    main()
