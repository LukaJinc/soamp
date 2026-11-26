"""
Dataset step 5: a linear-peptides-only copy of an already built dataset.

Filters a built dataset (default: data/dev_v2, the seven-strain data) down to the peptides that
are linear -- no disulfide bridge, no head-to-tail or side-chain cyclization -- and writes the
result as its own dataset directory, so the two can be trained side by side. Nothing is
re-labelled and no split is recomputed: every surviving (peptide, strain) pair keeps the split
and fold it has in the source dataset, so a dev_v2_linear result is comparable to the dev_v2
result of the same cell apart from the rows that are gone.

The linearity rule and why it needs both the DBAASP flag and the SMILES are in
``soamp.data.linear_subset``.

    python pipeline/data/05_build_linear_subset.py --config config/data/dev_v2_linear.yaml

Writes into data/dev_v2_linear/: mic_classification_dataset.csv, peptide_split.csv, the
per-peptide feature tables that exist in the source dir, and the organism vocab (copied). The
peptide feature scalers are NOT written here -- refit them on this dataset's train rows:

    python pipeline/features/03_fit_peptide_scaler.py --config config/features/dev_v2_linear_peptide_rdkit.yaml
    python pipeline/features/03_fit_peptide_scaler.py --config config/features/dev_v2_linear_peptide_peptideclm.yaml
    python pipeline/features/03_fit_peptide_scaler.py --config config/features/dev_v2_linear_peptide_molgraph.yaml

Audit files in reports/dev_v2_linear/: peptide_linearity.csv (per peptide: both flags and the
drop reason), linear_subset_per_strain.csv (pairs kept/dropped and active share per strain).
"""
import argparse
import shutil

import pandas as pd
from dotenv import load_dotenv

from soamp.data.linear_subset import LinearSubsetConfig, linear_subset, subset_feature_csv
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data/dev_v2_linear.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, LinearSubsetConfig)
    log = configure_logging("build_linear_subset")

    source = cfg.resolve(cfg.paths.source_dir)
    out_dir = cfg.resolve(cfg.paths.out_dir)
    reports = cfg.resolve(cfg.paths.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(source / cfg.dataset_filename)
    peptide_split = pd.read_csv(source / cfg.peptide_split_filename)
    result = linear_subset(dataset, peptide_split)
    linear_ids = set(result.dataset["peptide_id"])

    result.dataset.to_csv(out_dir / cfg.dataset_filename, index=False)
    result.peptide_split.to_csv(out_dir / cfg.peptide_split_filename, index=False)
    result.linearity.to_csv(reports / cfg.linearity_filename, index=False)
    result.per_strain.to_csv(reports / cfg.per_strain_filename)
    written = [out_dir / cfg.dataset_filename, out_dir / cfg.peptide_split_filename]

    for name in cfg.feature_files:
        src = source / name
        if not src.is_file():
            log.warning("feature table absent in %s, skipped: %s", source, name)
            continue
        subset_feature_csv(pd.read_csv(src), linear_ids).to_csv(out_dir / name, index=False)
        written.append(out_dir / name)
    for name in cfg.copy_files:
        src = source / name
        if not src.is_file():
            log.warning("file absent in %s, skipped: %s", source, name)
            continue
        shutil.copyfile(src, out_dir / name)
        written.append(out_dir / name)

    lines = ["=== Linear-peptides-only subset ===", f"source: {source}",
             *(f"{k}: {v}" for k, v in result.stats.items()), "",
             "Per strain:", result.per_strain.to_string(), "",
             *(f"Wrote {p}" for p in written),
             f"Wrote {reports / cfg.linearity_filename}", f"Wrote {reports / cfg.per_strain_filename}",
             "", "Next: refit the peptide scalers (see this script's docstring) and publish the data",
             "artifact with scripts/exp/publish_data.py --dataset <name>."]
    (reports / cfg.summary_filename).write_text("\n".join(lines) + "\n")
    log.info("\n".join(lines))


if __name__ == "__main__":
    main()
