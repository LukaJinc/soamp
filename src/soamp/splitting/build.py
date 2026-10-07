"""Top-level orchestration for the peptide split pipeline stage.

Builds a union similarity graph (Morgan/ECFP Tanimoto fingerprint edges OR
QMAP BLOSUM45 sequence-identity edges), clusters it with QMAP's own Leiden
community detection (reused, not reimplemented), then carves the resulting
clusters into a train/test/CV-fold split via
soamp.splitting.bucketing.stratified_cluster_split, which balances
bucket size and the dataset-wide has_noncanonical and active proportions.
"""
from dataclasses import dataclass

import pandas as pd
from qmap.toolkit.clustering.community_detection import leiden_community_detection

from soamp.splitting.bucketing import stratified_cluster_split
from soamp.splitting.graph import (
    compute_fingerprint_edges,
    compute_qmap_edges,
    contract_hard_links,
    duplicate_pairs,
    is_qmap_scoreable,
)


class SplitBuildError(ValueError):
    """Raised by unique_peptides_for_splitting on a peptide_id with
    conflicting sequence/smiles/has_noncanonical values across its rows."""


def unique_peptides_for_splitting(rows: list[dict]) -> list[dict]:
    """Dedups mic_classification_dataset.csv-shaped rows (repeated once per
    organism a peptide was tested against) to one
    {'peptide_id', 'sequence', 'smiles', 'has_noncanonical', 'n_rows',
    'n_active'} dict per peptide_id, order-preserving on first occurrence.
    n_rows counts the peptide's (peptide, organism) records; n_active counts
    those whose `label` is "active" (0 if rows carry no `label`). has_noncanonical is
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
            seen[pid]["n_rows"] += 1
            seen[pid]["n_active"] += int(row.get("label") == "active")
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
        candidate["n_rows"] = 1
        candidate["n_active"] = int(row.get("label") == "active")
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
    hard_link_fingerprint_threshold: float = 0.95,
    hard_link_identity_threshold: float = 0.90,
    leiden_n_iterations: int = 2,
    leiden_seed: int = 42,
    test_size: float = 0.2,
    n_folds: int = 5,
    bucket_lambda_noncanonical: float = 8.0,
    bucket_lambda_active: float = 0.0,
    bucket_n_iterations: int = 40_000,
    bucket_seed: int = 42,
) -> SplitBuildResult:
    """Builds the fingerprint+QMAP union graph, clusters it, and carves a
    stratified train/test/CV-fold split. `peptides` must already be deduped
    to one dict per peptide_id (see unique_peptides_for_splitting). If every
    peptide carries `n_rows`/`n_active`, bucket size and the non-canonical /
    active proportions are balanced over (peptide, organism) records;
    otherwise over unique peptides, with no active term.

    Leakage rule: Leiden communities alone do not stop near-identical peptides
    from landing in different buckets (the union graph is one giant connected
    component, so communities necessarily cut edges). Peptides that are
    duplicates (identical SMILES, or identical non-'X' sequence) or
    near-duplicates (Tanimoto >= hard_link_fingerprint_threshold, or QMAP
    identity >= hard_link_identity_threshold) are contracted into a single
    node before Leiden runs, so they always share a community and therefore
    a bucket."""
    peptide_ids = [p["peptide_id"] for p in peptides]
    sequences = [p["sequence"] for p in peptides]
    smiles = [p["smiles"] for p in peptides]
    has_noncanonical = [p["has_noncanonical"] for p in peptides]
    n = len(peptides)
    has_records = n > 0 and all("n_rows" in p and "n_active" in p for p in peptides)
    row_counts = [p["n_rows"] for p in peptides] if has_records else None
    active_counts = [p["n_active"] for p in peptides] if has_records else None

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
    hard_pairs = sorted(
        set(duplicate_pairs(sequences, smiles))
        | {e for e, sim in fingerprint_edges.items() if sim >= hard_link_fingerprint_threshold}
        | {e for e, ident in qmap_edges.items() if ident >= hard_link_identity_threshold}
    )
    graph, node_of_peptide = contract_hard_links(n, hard_pairs, fingerprint_edges, qmap_edges)
    group_communities = leiden_community_detection(
        graph, n_iterations=leiden_n_iterations, seed=leiden_seed
    )

    # Leiden may omit isolated groups; give each its own community, then map
    # group communities back onto peptides.
    community_of_group = dict(
        zip(group_communities["node_id"], group_communities["community"])
    )
    next_community = (max(community_of_group.values()) + 1) if community_of_group else 0
    for group in range(graph.vcount()):
        if group not in community_of_group:
            community_of_group[group] = next_community
            next_community += 1
    node_communities = pd.DataFrame(
        {"node_id": range(n), "community": [community_of_group[g] for g in node_of_peptide]}
    )

    assignment = stratified_cluster_split(
        node_communities, peptide_ids, has_noncanonical,
        row_counts=row_counts, active_counts=active_counts,
        test_size=test_size, n_folds=n_folds,
        lambda_noncanonical=bucket_lambda_noncanonical,
        lambda_active=bucket_lambda_active,
        n_iterations=bucket_n_iterations, seed=bucket_seed,
    )

    communities = [assignment.community_by_peptide_id[pid] for pid in peptide_ids]
    n_clusters = len(set(communities))
    largest_cluster_fraction = (
        float(pd.Series(communities).value_counts().max()) / n if n else 0.0
    )
    n_qmap_unscoreable = sum(1 for seq in sequences if not is_qmap_scoreable(seq))
    dataset_nc_fraction = (sum(has_noncanonical) / n) if n else 0.0
    dataset_record_stats = {}
    if has_records:
        total_rows = sum(row_counts)
        dataset_record_stats = {
            "dataset_n_rows": total_rows,
            "dataset_active_fraction_rows": sum(active_counts) / total_rows,
            "dataset_has_noncanonical_fraction_rows": sum(
                r for r, nc in zip(row_counts, has_noncanonical) if nc
            ) / total_rows,
        }

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

    bucket_of = [
        assignment.split_by_peptide_id[pid]
        if assignment.fold_by_peptide_id[pid] is None
        else assignment.fold_by_peptide_id[pid]
        for pid in peptide_ids
    ]
    cross_bucket = {
        "fingerprint": sum(bucket_of[i] != bucket_of[j] for i, j in fingerprint_edges),
        "qmap": sum(bucket_of[i] != bucket_of[j] for i, j in qmap_edges),
        "hard_links": sum(bucket_of[i] != bucket_of[j] for i, j in hard_pairs),
    }

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
        "hard_link_fingerprint_threshold": hard_link_fingerprint_threshold,
        "hard_link_identity_threshold": hard_link_identity_threshold,
        "n_hard_links": len(hard_pairs),
        "n_hard_link_groups": graph.vcount(),
        "cross_bucket_edges": {
            **cross_bucket,
            "fingerprint_total": len(fingerprint_edges),
            "qmap_total": len(qmap_edges),
        },
        "leiden_n_iterations": leiden_n_iterations,
        "leiden_seed": leiden_seed,
        "test_size": test_size,
        "n_folds": n_folds,
        "bucket_lambda_noncanonical": bucket_lambda_noncanonical,
        "bucket_lambda_active": bucket_lambda_active,
        "balanced_over": "records" if has_records else "peptides",
        "bucket_n_iterations": bucket_n_iterations,
        "bucket_seed": bucket_seed,
        "n_clusters": n_clusters,
        "largest_cluster_fraction": largest_cluster_fraction,
        "dataset_has_noncanonical_fraction": dataset_nc_fraction,
        "n_qmap_unscoreable_peptides": n_qmap_unscoreable,
        **dataset_record_stats,
        "bucket_stats": assignment.bucket_stats,
    }

    return SplitBuildResult(rows=rows, sidecar=sidecar)
