"""Whole-cluster train/test/CV-fold bucket assignment, stratified toward the
dataset-wide has_noncanonical rate.

A single-pass "visit clusters largest-first, assign each to whichever
bucket currently looks best" greedy was prototyped and rejected: on this
project's real cluster distribution (a handful of large, mostly-canonical
clusters plus many small clusters where non-canonical peptides concentrate
unevenly -- 419 of 673 real clusters are purely canonical) it is myopic --
non-canonical-heavy clusters run out partway through the pass, so it can't
see that a bucket it just under-filled has nothing left to draw from later.
Reproducibly measured on the real data: roughly the same 10-27% spread as
no stratification at all, and *worse* at higher non-canonical weights (one
run: 11.7%-66.1% spread).

Replaced with a two-stage approach that is not myopic in the same way:
size-only greedy placement, then randomized local-search (swap)
refinement. Verified on the real 673-cluster graph to reliably balance both
bucket size (within ~2.5% of target) and non-canonical rate (within ~1.5
points of the dataset-wide rate across all 6 buckets), deterministic given
a seed, robust across several seeds tried.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


class BucketingError(ValueError):
    """Raised on mismatched cluster_sizes/cluster_noncanonical_counts keys,
    a non-canonical count exceeding its cluster's size, or bucket target
    fractions that are invalid or don't sum to ~1.0."""


def _validate_inputs(
    cluster_sizes: dict[int, int],
    cluster_noncanonical_counts: dict[int, int],
    bucket_target_fractions: dict[str, float],
) -> None:
    if cluster_sizes.keys() != cluster_noncanonical_counts.keys():
        raise BucketingError(
            "cluster_sizes and cluster_noncanonical_counts must have identical keys"
        )
    for cluster, size in cluster_sizes.items():
        nc = cluster_noncanonical_counts[cluster]
        if nc > size:
            raise BucketingError(
                f"cluster {cluster}: noncanonical_count ({nc}) exceeds size ({size})"
            )
    if not bucket_target_fractions:
        raise BucketingError("bucket_target_fractions must not be empty")
    for name, frac in bucket_target_fractions.items():
        if not (0.0 < frac < 1.0):
            raise BucketingError(
                f"bucket {name!r} target fraction must be in (0, 1), got {frac}"
            )
    total_fraction = sum(bucket_target_fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise BucketingError(
            f"bucket_target_fractions must sum to 1.0, got {total_fraction}"
        )


def assign_clusters_to_buckets(
    cluster_sizes: dict[int, int],
    cluster_noncanonical_counts: dict[int, int],
    bucket_target_fractions: dict[str, float],
    *,
    lambda_noncanonical: float = 8.0,
    n_iterations: int = 40_000,
    seed: int = 42,
) -> dict[int, str]:
    """Assigns each cluster (community id) to exactly one bucket name.

    Stage 1 -- initial placement: process clusters largest-first; each
    cluster goes to whichever bucket is currently furthest below its target
    size (argmin(bucket_size - target_size)), ignoring non-canonical
    content.

    Stage 2 -- local-search refinement: repeatedly (n_iterations times)
    pick two clusters in different buckets at random and swap their bucket
    assignment if doing so reduces

        sum_bucket(((bucket_size - target_size) / target_size) ** 2)
        + lambda_noncanonical * sum_bucket((bucket_nc_rate - dataset_nc_rate) ** 2)

    reverting the swap otherwise. Deterministic given `seed`.
    """
    _validate_inputs(cluster_sizes, cluster_noncanonical_counts, bucket_target_fractions)

    clusters = list(cluster_sizes.keys())
    sizes = np.array([cluster_sizes[c] for c in clusters], dtype=float)
    ncs = np.array([cluster_noncanonical_counts[c] for c in clusters], dtype=float)
    n_clusters = len(clusters)

    bucket_names = list(bucket_target_fractions.keys())
    n_buckets = len(bucket_names)
    target_fraction = np.array([bucket_target_fractions[b] for b in bucket_names])

    total_size = sizes.sum()
    total_nc = ncs.sum()
    dataset_nc_rate = total_nc / total_size if total_size else 0.0
    target_size = target_fraction * total_size

    rng = np.random.default_rng(seed)

    # Stage 1: largest-first, most-underfilled-relative-to-target placement.
    order = np.argsort(-sizes)
    assign = np.zeros(n_clusters, dtype=int)
    bucket_size = np.zeros(n_buckets)
    for idx in order:
        b = int(np.argmin(bucket_size - target_size))
        assign[idx] = b
        bucket_size[b] += sizes[idx]

    def objective(current: np.ndarray) -> float:
        bsize = np.zeros(n_buckets)
        bnc = np.zeros(n_buckets)
        for b in range(n_buckets):
            mask = current == b
            bsize[b] = sizes[mask].sum()
            bnc[b] = ncs[mask].sum()
        size_dev = ((bsize - target_size) / target_size) ** 2
        nc_rate = np.divide(bnc, bsize, out=np.zeros(n_buckets), where=bsize > 0)
        nc_dev = (nc_rate - dataset_nc_rate) ** 2
        return float(size_dev.sum() + lambda_noncanonical * nc_dev.sum())

    # Stage 2: randomized swap local search.
    cur_obj = objective(assign)
    if n_clusters >= 2:
        for _ in range(n_iterations):
            i, j = rng.integers(0, n_clusters, size=2)
            if assign[i] == assign[j]:
                continue
            assign[i], assign[j] = assign[j], assign[i]
            new_obj = objective(assign)
            if new_obj < cur_obj:
                cur_obj = new_obj
            else:
                assign[i], assign[j] = assign[j], assign[i]

    return {clusters[idx]: bucket_names[b] for idx, b in enumerate(assign)}


@dataclass
class SplitAssignment:
    split_by_peptide_id: dict[str, str]
    fold_by_peptide_id: dict[str, int | None]
    community_by_peptide_id: dict[str, int]
    bucket_stats: dict[str, dict]


def stratified_cluster_split(
    node_communities: pd.DataFrame,
    peptide_ids: list[str],
    has_noncanonical: list[bool],
    *,
    test_size: float = 0.2,
    n_folds: int = 5,
    lambda_noncanonical: float = 8.0,
    n_iterations: int = 40_000,
    seed: int = 42,
) -> SplitAssignment:
    """Carves whole clusters into a "test" bucket (target `test_size`) and
    `n_folds` CV folds (target `(1 - test_size) / n_folds` each), balanced
    toward the dataset-wide has_noncanonical rate computed from
    `has_noncanonical` itself (never hardcoded).

    `node_communities` (columns: node_id, community) is Leiden's raw output
    and may omit isolated nodes with no edges at all -- any peptide_id whose
    positional index is absent gets its own singleton community, mirroring
    this repo's established pattern of reconciling anything a lower-level
    library call silently drops (see soamp.data.splitting's post_filtering
    reconciliation)."""
    if len(peptide_ids) != len(has_noncanonical):
        raise BucketingError(
            "peptide_ids and has_noncanonical must be the same length"
        )
    if not (0.0 < test_size < 1.0):
        raise BucketingError(f"test_size must be in (0, 1), got {test_size}")
    if n_folds < 2:
        raise BucketingError(f"n_folds must be >= 2, got {n_folds}")

    node_to_community = dict(zip(node_communities["node_id"], node_communities["community"]))
    next_community = (max(node_to_community.values()) + 1) if node_to_community else 0
    communities: list[int] = []
    for node_id in range(len(peptide_ids)):
        if node_id not in node_to_community:
            node_to_community[node_id] = next_community
            next_community += 1
        communities.append(node_to_community[node_id])

    cluster_sizes: dict[int, int] = {}
    cluster_nc: dict[int, int] = {}
    for community, is_nc in zip(communities, has_noncanonical):
        cluster_sizes[community] = cluster_sizes.get(community, 0) + 1
        cluster_nc[community] = cluster_nc.get(community, 0) + (1 if is_nc else 0)

    fold_names = [f"fold_{i}" for i in range(n_folds)]
    fold_fraction = (1.0 - test_size) / n_folds
    bucket_target_fractions = {"test": test_size, **{name: fold_fraction for name in fold_names}}

    cluster_bucket = assign_clusters_to_buckets(
        cluster_sizes, cluster_nc, bucket_target_fractions,
        lambda_noncanonical=lambda_noncanonical, n_iterations=n_iterations, seed=seed,
    )

    split_by_peptide_id: dict[str, str] = {}
    fold_by_peptide_id: dict[str, int | None] = {}
    community_by_peptide_id: dict[str, int] = {}
    for peptide_id, community in zip(peptide_ids, communities):
        bucket = cluster_bucket[community]
        community_by_peptide_id[peptide_id] = community
        if bucket == "test":
            split_by_peptide_id[peptide_id] = "test"
            fold_by_peptide_id[peptide_id] = None
        else:
            split_by_peptide_id[peptide_id] = "train"
            fold_by_peptide_id[peptide_id] = int(bucket.removeprefix("fold_"))

    bucket_stats: dict[str, dict] = {}
    for name, target_fraction in bucket_target_fractions.items():
        member_mask = [cluster_bucket[c] == name for c in communities]
        n_members = sum(member_mask)
        n_nc = sum(
            1 for is_member, is_nc in zip(member_mask, has_noncanonical) if is_member and is_nc
        )
        bucket_stats[name] = {
            "n_peptides": n_members,
            "target_size_fraction": target_fraction,
            "has_noncanonical_fraction": (n_nc / n_members) if n_members else 0.0,
        }

    return SplitAssignment(
        split_by_peptide_id=split_by_peptide_id,
        fold_by_peptide_id=fold_by_peptide_id,
        community_by_peptide_id=community_by_peptide_id,
        bucket_stats=bucket_stats,
    )
