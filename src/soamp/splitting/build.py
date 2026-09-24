"""Top-level orchestration for the peptide split pipeline stage.

Builds a union similarity graph (Morgan/ECFP Tanimoto fingerprint edges OR
QMAP BLOSUM45 sequence-identity edges), clusters it with QMAP's own Leiden
community detection (reused, not reimplemented), then carves the resulting
clusters into a train/test/CV-fold split via
soamp.splitting.bucketing.stratified_cluster_split, which balances both
bucket size and the dataset-wide has_noncanonical rate.
"""
from dataclasses import dataclass

import pandas as pd
from qmap.toolkit.clustering.community_detection import leiden_community_detection

from soamp.splitting.bucketing import stratified_cluster_split
from soamp.splitting.graph import (
    build_union_graph,
    compute_fingerprint_edges,
    compute_qmap_edges,
    is_qmap_scoreable,
)


class SplitBuildError(ValueError):
    """Raised by unique_peptides_for_splitting on a peptide_id with
    conflicting sequence/smiles/has_noncanonical values across its rows."""


def unique_peptides_for_splitting(rows: list[dict]) -> list[dict]:
    """Dedups mic_classification_dataset.csv-shaped rows (repeated once per
    organism a peptide was tested against) to one
    {'peptide_id', 'sequence', 'smiles', 'has_noncanonical'} dict per
    peptide_id, order-preserving on first occurrence. has_noncanonical is
    parsed via `value == "True"` (matches
    pipeline/data/02_split_train_validation.py's convention -- NOT
    bool(value), which would treat the string "False" as truthy). Raises
    SplitBuildError on conflicting sequence/smiles/has_noncanonical values
    for the same peptide_id."""
    seen: dict[str, dict] = {}
    out: list[dict] = []
    for row in rows:
        pid = row["peptide_id"]
        candidate = {
            "peptide_id": pid,
            "sequence": row["sequence"],
            "smiles": row["smiles"],
            "has_noncanonical": row["has_noncanonical"] == "True",
        }
        if pid in seen:
            prior = seen[pid]
            if (
                prior["sequence"] != candidate["sequence"]
                or prior["smiles"] != candidate["smiles"]
                or prior["has_noncanonical"] != candidate["has_noncanonical"]
            ):
                raise SplitBuildError(
                    f"peptide_id {pid!r} has conflicting sequence/smiles/has_noncanonical "
                    f"values across rows: {prior} vs {candidate}"
                )
            continue
        seen[pid] = candidate
        out.append(candidate)
    return out


@dataclass
class SplitBuildResult:
    rows: list[dict]
    sidecar: dict


def build_peptide_split(
    peptides: list[dict],
    *,
    fingerprint_threshold: float = 0.8,
    fingerprint_radius: int = 2,
    fingerprint_n_bits: int = 2048,
    identity_threshold: float = 0.6,
    identity_matrix: str = "blosum45",
    identity_gap_open: int = 5,
    identity_gap_extension: int = 1,
    identity_use_cache: bool = True,
    identity_show_progress: bool = True,
    identity_num_threads: int | None = None,
    leiden_n_iterations: int = 2,
    leiden_seed: int = 42,
    test_size: float = 0.2,
    n_folds: int = 5,
    bucket_lambda_noncanonical: float = 8.0,
    bucket_n_iterations: int = 40_000,
    bucket_seed: int = 42,
) -> SplitBuildResult:
    """Builds the fingerprint+QMAP union graph, clusters it, and carves a
    stratified train/test/CV-fold split. `peptides` must already be deduped
    to one dict per peptide_id (see unique_peptides_for_splitting)."""
    peptide_ids = [p["peptide_id"] for p in peptides]
    sequences = [p["sequence"] for p in peptides]
    smiles = [p["smiles"] for p in peptides]
    has_noncanonical = [p["has_noncanonical"] for p in peptides]
    n = len(peptides)

    fingerprint_edges = compute_fingerprint_edges(
        peptide_ids, smiles, fingerprint_threshold,
        radius=fingerprint_radius, n_bits=fingerprint_n_bits,
    )
    qmap_edges = compute_qmap_edges(
        peptide_ids, sequences, identity_threshold,
        matrix=identity_matrix, gap_open=identity_gap_open,
        gap_extension=identity_gap_extension, use_cache=identity_use_cache,
        show_progress=identity_show_progress, num_threads=identity_num_threads,
    )
    graph = build_union_graph(n, fingerprint_edges, qmap_edges)
    node_communities = leiden_community_detection(
        graph, n_iterations=leiden_n_iterations, seed=leiden_seed
    )

    assignment = stratified_cluster_split(
        node_communities, peptide_ids, has_noncanonical,
        test_size=test_size, n_folds=n_folds,
        lambda_noncanonical=bucket_lambda_noncanonical,
        n_iterations=bucket_n_iterations, seed=bucket_seed,
    )

    communities = [assignment.community_by_peptide_id[pid] for pid in peptide_ids]
    n_clusters = len(set(communities))
    largest_cluster_fraction = (
        float(pd.Series(communities).value_counts().max()) / n if n else 0.0
    )
    n_qmap_unscoreable = sum(1 for seq in sequences if not is_qmap_scoreable(seq))
    dataset_nc_fraction = (sum(has_noncanonical) / n) if n else 0.0

    rows = [
        {
            "peptide_id": pid,
            "community": assignment.community_by_peptide_id[pid],
            "split": assignment.split_by_peptide_id[pid],
            "fold_id": assignment.fold_by_peptide_id[pid],
            "has_noncanonical": has_nc,
        }
        for pid, has_nc in zip(peptide_ids, has_noncanonical)
    ]

    sidecar = {
        "method": (
            "union graph (Morgan/ECFP Tanimoto >= fingerprint_threshold OR QMAP "
            "BLOSUM45 identity >= identity_threshold, both computed once and "
            "unioned) -> qmap.toolkit leiden_community_detection -> stratified "
            "whole-cluster bucket assignment (test + n_folds CV folds, balanced "
            "toward the dataset-wide has_noncanonical rate)"
        ),
        "n_peptides": n,
        "fingerprint_threshold": fingerprint_threshold,
        "fingerprint_radius": fingerprint_radius,
        "fingerprint_n_bits": fingerprint_n_bits,
        "identity_threshold": identity_threshold,
        "identity_matrix": identity_matrix,
        "identity_gap_open": identity_gap_open,
        "identity_gap_extension": identity_gap_extension,
        "leiden_n_iterations": leiden_n_iterations,
        "leiden_seed": leiden_seed,
        "test_size": test_size,
        "n_folds": n_folds,
        "bucket_lambda_noncanonical": bucket_lambda_noncanonical,
        "bucket_n_iterations": bucket_n_iterations,
        "bucket_seed": bucket_seed,
        "n_clusters": n_clusters,
        "largest_cluster_fraction": largest_cluster_fraction,
        "dataset_has_noncanonical_fraction": dataset_nc_fraction,
        "n_qmap_unscoreable_peptides": n_qmap_unscoreable,
        "bucket_stats": assignment.bucket_stats,
    }

    return SplitBuildResult(rows=rows, sidecar=sidecar)
