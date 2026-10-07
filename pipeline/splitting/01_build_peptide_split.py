"""
Splitting step 1: build the peptide train/test/CV-fold split.

Reads data/mic_classification_dataset.csv, dedups to one row per unique
peptide_id, builds a union similarity graph (Morgan/ECFP Tanimoto
fingerprint edges OR QMAP BLOSUM45 sequence-identity edges -- see
soamp.splitting.graph), clusters it via QMAP's own Leiden community
detection, then carves the clusters into a train/test/CV-fold split
stratified toward the dataset-wide has_noncanonical rate (see
soamp.splitting.bucketing). Replaces the exploratory
scripts/EDA/01-05 notebooks' 4-way consensus-clustering approach --
two of those four voters (RDKit-descriptor and PeptideCLM k-means) had
near-zero pairwise agreement with everything else, so this stage unions
only the two genuine near-duplicate detectors (fingerprint + QMAP). Fills
CLAUDE.md's "Stage 2 -- Splitting: not yet its own module" gap.

Output: data/<output.filename> (peptide_id, community, split, fold_id,
has_noncanonical) and data/<output.sidecar_filename> (method + every
parameter used + resulting cluster/bucket diagnostics, for reproducibility
-- mirrors data/val_split.json's shape).
"""
import argparse
import csv
import json

from dotenv import load_dotenv

from soamp.splitting.build import build_peptide_split, unique_peptides_for_splitting
from soamp.splitting.config import SplittingConfig
from soamp.utils.config import load_config
from soamp.utils.logging import configure_logging

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/splitting/base.yaml")
    return parser.parse_args()


ARGS = _parse_args()
CFG = load_config(ARGS.config, SplittingConfig)
CLASSIFICATION_CSV = CFG.paths.data_dir / CFG.input_files.classification_dataset_filename
OUT_CSV = CFG.paths.data_dir / CFG.output.filename
OUT_JSON = CFG.paths.data_dir / CFG.output.sidecar_filename


def main() -> None:
    log = configure_logging("splitting.01_build_peptide_split")

    with open(CLASSIFICATION_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    peptides = unique_peptides_for_splitting(rows)
    log.info(f"{len(rows)} classification rows -> {len(peptides)} unique peptides")

    result = build_peptide_split(
        peptides,
        fingerprint_threshold=CFG.fingerprint_graph.fingerprint_threshold,
        fingerprint_radius=CFG.fingerprint_graph.radius,
        fingerprint_n_bits=CFG.fingerprint_graph.n_bits,
        identity_threshold=CFG.qmap_graph.identity_threshold,
        identity_matrix=CFG.qmap_graph.matrix,
        identity_gap_open=CFG.qmap_graph.gap_open,
        identity_gap_extension=CFG.qmap_graph.gap_extension,
        identity_use_cache=CFG.qmap_graph.use_cache,
        identity_num_threads=CFG.qmap_graph.num_threads,
        hard_link_fingerprint_threshold=CFG.hard_links.fingerprint_threshold,
        hard_link_identity_threshold=CFG.hard_links.identity_threshold,
        leiden_n_iterations=CFG.leiden.n_iterations,
        leiden_seed=CFG.leiden.seed,
        test_size=CFG.bucketing.test_size,
        n_folds=CFG.bucketing.n_folds,
        bucket_lambda_noncanonical=CFG.bucketing.lambda_noncanonical,
        bucket_lambda_active=CFG.bucketing.lambda_active,
        bucket_n_iterations=CFG.bucketing.n_iterations,
        bucket_seed=CFG.bucketing.seed,
    )

    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["peptide_id", "community", "split", "fold_id", "has_noncanonical"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.rows)

    with open(OUT_JSON, "w") as f:
        json.dump(result.sidecar, f, indent=2)

    log.info(
        f"n_clusters={result.sidecar['n_clusters']} "
        f"largest_cluster_fraction={result.sidecar['largest_cluster_fraction']:.3f} "
        f"dataset_has_noncanonical_fraction={result.sidecar['dataset_has_noncanonical_fraction']:.3f} "
        f"balanced_over={result.sidecar['balanced_over']}"
    )
    log.info(
        f"hard links: {result.sidecar['n_hard_links']} pairs -> "
        f"{result.sidecar['n_hard_link_groups']} contracted nodes; "
        f"cross-bucket edges: {result.sidecar['cross_bucket_edges']}"
    )
    for name, stats in result.sidecar["bucket_stats"].items():
        log.info(f"bucket {name}: {stats}")
    log.info(f"Wrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
