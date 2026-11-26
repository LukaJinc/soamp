"""
Dataset step 3: classification dataset + CV split from the dev DBAASP data.

Two modes, chosen by the config (one entrypoint, no second loop to drift):

* ``config/data/dev.yaml`` (default, ``paths.cluster_parquet`` unset): reads
  data/dev/dbaasp_dev_processed.parquet and the ready-made data/dev/dbaasp_dev_split.parquet
  (scripts/EDA/09_data_processing.ipynb), labels MIC records active/inactive from `activity`
  with the shared organism thresholds (strain -> species), drops uncertain/conflicting pairs,
  and writes into data/dev/:
    mic_classification_dataset.csv   (same columns as the legacy dataset)
    peptide_split.csv                (peptide_id, community, split, fold_id, has_noncanonical)

* ``config/data/dev_v2.yaml`` (``paths.cluster_parquet`` set): the seven-strain dataset.
  Reads the table written by pipeline/data/04_recompute_mic.py (MIC recomputed in ug/ml, NA
  rows kept at 1,111,111), labels the pairs (25 / 100 ug/ml), THEN builds the test + 5-fold
  split from the precomputed clusters with the stratified search, weighted by the labelled
  pairs (size, non-linear share, active share), and writes into data/dev_v2/:
    mic_classification_dataset.csv, peptide_split.csv       (same schemas as above)
    dbaasp_dev_split_final.parquet, dev_split_final.json    (the split and its parameters)
  plus audit files in reports/dev_v2/ (NA-only pairs, label conflicts, build log).

Training configs use `paths.data_dir: data/dev` or `data/dev_v2` to pick these up.
"""
import argparse
import json

import pandas as pd
from dotenv import load_dotenv

from soamp.common.thresholds import load_threshold_table
from soamp.data.dev_dataset import DevDatasetConfig, attach_split, build_dev_dataset, label_dev_pairs
from soamp.splitting.dev_final_split import check_final_split, split_labelled_peptides
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
    table = load_threshold_table(cfg.resolve(cfg.paths.thresholds_csv))
    out_dir = cfg.resolve(cfg.paths.out_dir)
    reports = cfg.resolve(cfg.paths.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    extra_lines: list[str] = []

    if cfg.paths.cluster_parquet is None:
        split = pd.read_parquet(cfg.resolve(cfg.paths.split_parquet))
        result = build_dev_dataset(processed, split, table, cfg)
        dataset, peptide_split, stats = result.dataset, result.peptide_split, result.stats
    else:
        labelled = label_dev_pairs(processed, table, cfg)
        clusters = pd.read_parquet(cfg.resolve(cfg.paths.cluster_parquet))
        final = split_labelled_peptides(labelled.dataset, clusters, cfg.bucketing)
        check_final_split(final)
        dataset, peptide_split = attach_split(labelled.dataset, final.table)
        stats = {**labelled.stats, "by_split": dataset["split"].value_counts().to_dict()}

        split_path = cfg.resolve(cfg.paths.split_parquet)
        final.table.to_parquet(split_path, index=False)
        (out_dir / cfg.split_sidecar_filename).write_text(json.dumps(final.sidecar, indent=1, default=str))
        labelled.na_only.to_csv(reports / cfg.na_only_filename, index=False)
        labelled.conflicts.to_csv(reports / cfg.conflicts_filename, index=False)

        buckets = final.table.set_index("peptide_id")
        dataset["bucket"] = dataset["peptide_id"].map(
            lambda p: "test" if buckets.at[p, "split"] == "test" else f"fold_{int(buckets.at[p, 'fold_id'])}")
        per_org = (dataset.assign(active=dataset["label"].eq("active"))
                          .pivot_table(index="organism", columns="bucket", values="active", aggfunc="mean") * 100).round(1)
        dataset = dataset.drop(columns="bucket")
        organism_lines = [line for name, tab in final.organism_tables.items()
                          for line in ("", f"Per organism x bucket, {name}:", tab.to_string())]
        extra_lines = ["", "Bucket summary (labelled pairs):", final.summary.to_string(), "",
                       "Active % per organism and bucket:", per_org.to_string(), *organism_lines,
                       f"Wrote {split_path}", f"Wrote {out_dir / cfg.split_sidecar_filename}",
                       f"Wrote {reports / cfg.na_only_filename} ({len(labelled.na_only)} pairs)",
                       f"Wrote {reports / cfg.conflicts_filename} ({len(labelled.conflicts)} pairs)"]

    dataset.to_csv(out_dir / cfg.dataset_filename, index=False)
    peptide_split.to_csv(out_dir / cfg.peptide_split_filename, index=False)

    lines = ["=== Dev classification dataset ===", *(f"{k}: {v}" for k, v in stats.items()),
             f"Wrote {out_dir / cfg.dataset_filename}", f"Wrote {out_dir / cfg.peptide_split_filename}", *extra_lines]
    (reports / cfg.summary_filename).write_text("\n".join(lines) + "\n")
    log.info("\n".join(lines))


if __name__ == "__main__":
    main()
