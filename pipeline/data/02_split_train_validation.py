"""
Dataset step 2: materialize the train/validation split used for
per-epoch monitoring and best-checkpoint selection.

Reads data/mic_classification_dataset.csv (dataset_model_ready), filters
to split=='train' rows, and carves out a validation subset by unique
peptide_id via soamp.data.splitting.split_train_validation -- the same
sequence-identity graph + Leiden community detection mechanism the
train/test split uses (pipeline/curation/07_build_final_and_split.py),
so that homologous peptides can't land one in 'fit' and one in 'val'. The
true 'test' split is never touched.

This is the single source of truth for the val split: pipeline/train.py
reads data/val_split.json directly rather than recomputing it, so there
is exactly one place this split is derived (see CLAUDE.md's "two cooks"
note on why duplicated derivation logic is avoided in this repo).

Also logs the has_noncanonical distribution across fit/val/test as a
visibility/audit metric -- non-canonical/cyclic peptide recovery is this
project's differentiator, so it's worth knowing whether any split ends up
skewed relative to the others, even though the splitter itself has no
stratification hook to actively balance it.

Output: data/val_split.json (val_split_v1) -- fit_peptide_ids,
val_peptide_ids, val_fraction, seed, identity_threshold, post_filtering.
"""
import argparse
import csv
import json

from dotenv import load_dotenv

from soamp.data.config import DatasetConfig
from soamp.data.splitting import split_train_validation
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, DatasetConfig)
CLASSIFICATION_CSV = CFG.paths.data_dir / CFG.output.filename
VAL_SPLIT_JSON = CFG.paths.data_dir / CFG.output.val_split_filename
LOG_PATH = CFG.paths.reports_dir / "data_step2_split_train_validation_log.txt"


def _noncanonical_fraction(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    n_noncanonical = sum(1 for r in rows if r["has_noncanonical"] == "True")
    return n_noncanonical / len(rows)


def main() -> None:
    log = configure_logging("data.02_split_train_validation")

    with open(CLASSIFICATION_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    log.info(f"Loaded {len(all_rows)} rows, {len(train_rows)} in 'train' split")

    fit_rows, val_rows = split_train_validation(
        train_rows, CFG.val_split.val_fraction, CFG.val_split.seed,
        CFG.val_split.identity_threshold, CFG.val_split.post_filtering,
    )
    fit_ids = sorted({r["peptide_id"] for r in fit_rows})
    val_ids = sorted({r["peptide_id"] for r in val_rows})

    fit_noncanonical_frac = _noncanonical_fraction(fit_rows)
    val_noncanonical_frac = _noncanonical_fraction(val_rows)
    test_noncanonical_frac = _noncanonical_fraction(test_rows)

    val_split = {
        "method": "qmap.toolkit.train_test_split (sequence-identity graph + Leiden community "
                  "detection) -- same mechanism as the train/test split",
        "val_fraction": CFG.val_split.val_fraction,
        "seed": CFG.val_split.seed,
        "identity_threshold": CFG.val_split.identity_threshold,
        "post_filtering": CFG.val_split.post_filtering,
        "split_level": "unique peptide_id (all organism rows for a peptide "
                       "share its fit/val assignment)",
        "fit_peptide_ids": [str(x) for x in fit_ids],
        "val_peptide_ids": [str(x) for x in val_ids],
        "has_noncanonical_fraction": {
            "fit": fit_noncanonical_frac,
            "val": val_noncanonical_frac,
            "test": test_noncanonical_frac,
        },
    }

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(VAL_SPLIT_JSON, "w") as f:
        json.dump(val_split, f, indent=2)

    CFG.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "=== Data step 2: train/validation split ===",
        f"val_fraction={CFG.val_split.val_fraction} seed={CFG.val_split.seed} "
        f"identity_threshold={CFG.val_split.identity_threshold} "
        f"post_filtering={CFG.val_split.post_filtering}",
        f"fit peptide_ids: {len(fit_ids)}  val peptide_ids: {len(val_ids)}",
        f"fit rows: {len(fit_rows)}  val rows: {len(val_rows)}  test rows: {len(test_rows)}",
        f"has_noncanonical fraction -- fit: {fit_noncanonical_frac:.3f}  "
        f"val: {val_noncanonical_frac:.3f}  test: {test_noncanonical_frac:.3f}",
        f"Wrote {VAL_SPLIT_JSON}",
    ]
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("\n".join(log_lines))


if __name__ == "__main__":
    main()
