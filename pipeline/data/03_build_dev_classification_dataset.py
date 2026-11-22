"""
Dataset step 3: classification dataset + CV split from the dev DBAASP data.

Reads data/dev/dbaasp_dev_processed.parquet and dbaasp_dev_split.parquet
(scripts/EDA/09_data_processing.ipynb), labels MIC records active/inactive from
`activity` (uM) with the shared organism thresholds (strain -> species), drops
uncertain/conflicting pairs, and writes into data/dev/:
  mic_classification_dataset.csv   (same columns as the legacy dataset)
  peptide_split.csv                (peptide_id, community, split, fold_id, has_noncanonical)
Training configs use `paths.data_dir: data/dev` to pick these up.
"""
import argparse

import pandas as pd
from dotenv import load_dotenv

from soamp.common.thresholds import load_threshold_table
from soamp.data.dev_dataset import DevDatasetConfig, build_dev_dataset
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data/dev.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, DevDatasetConfig)
    log = configure_logging("build_dev_classification_dataset")

    processed = pd.read_parquet(cfg.resolve(cfg.paths.processed_parquet))
    split = pd.read_parquet(cfg.resolve(cfg.paths.split_parquet))
    table = load_threshold_table(cfg.resolve(cfg.paths.thresholds_csv))
    result = build_dev_dataset(processed, split, table, cfg)

    out_dir = cfg.resolve(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.dataset.to_csv(out_dir / cfg.dataset_filename, index=False)
    result.peptide_split.to_csv(out_dir / cfg.peptide_split_filename, index=False)

    lines = ["=== Dev classification dataset ===", *(f"{k}: {v}" for k, v in result.stats.items()),
             f"Wrote {out_dir / cfg.dataset_filename}", f"Wrote {out_dir / cfg.peptide_split_filename}"]
    reports = cfg.resolve(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / cfg.summary_filename).write_text("\n".join(lines) + "\n")
    log.info("\n".join(lines))


if __name__ == "__main__":
    main()
