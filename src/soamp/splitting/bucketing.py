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

Balance targets (size, non-canonical rate, active rate) are measured over
caller-supplied *weights* -- in this project, (peptide, organism) records,
because the active/inactive label lives on a record, not a peptide. Weights
only change how a cluster is *scored*; the unit that is moved is always a
whole cluster, so every record of a peptide (and every graph neighbour of
it) stays in the same bucket -- balancing can never introduce leakage.
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
    cluster_active_counts: dict[int, int] | None = None,
) -> None:
    if cluster_sizes.keys() != cluster_noncanonical_counts.keys():
        raise BucketingError(
            "cluster_sizes and cluster_noncanonical_counts must have identical keys"
        )
    if cluster_active_counts is not None and cluster_active_counts.keys() != cluster_sizes.keys():
        raise BucketingError("cluster_active_counts must have the same keys as cluster_sizes")
    for cluster, size in cluster_sizes.items():
        nc = cluster_noncanonical_counts[cluster]
        if nc > size:
            raise BucketingError(
                f"cluster {cluster}: noncanonical_count ({nc}) exceeds size ({size})"
            )
        if cluster_active_counts is not None and cluster_active_counts[cluster] > size:
            raise BucketingError(
                f"cluster {cluster}: active_count ({cluster_active_counts[cluster]}) "
                f"exceeds size ({size})"
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
    cluster_active_counts: dict[int, int] | None = None,
    lambda_noncanonical: float = 8.0,
    lambda_active: float = 0.0,
    n_iterations: int = 40_000,
    seed: int = 42,
) -> dict[int, str]:
    """Assigns each cluster (community id) to exactly one bucket name.

    `cluster_sizes` / `cluster_*_counts` are weights (peptides, or records
    when the caller wants record-level proportions); they must be in the same
    unit.

    Stage 1 -- initial placement: process clusters largest-first; each
    cluster goes to whichever bucket is currently furthest below its target
    size (argmin(bucket_size - target_size)), ignoring composition.

    Stage 2 -- local-search refinement: repeatedly (n_iterations times)
    pick two clusters in different buckets at random and swap their bucket
    assignment if doing so reduces

        sum_bucket(((bucket_size - target_size) / target_size) ** 2)
        + lambda_noncanonical * sum_bucket((bucket_nc_rate - dataset_nc_rate) ** 2)
        + lambda_active * sum_bucket((bucket_active_rate - dataset_active_rate) ** 2)

    reverting the swap otherwise. The active term is skipped when
    `cluster_active_counts` is None. Deterministic given `seed`.
    """
    _validate_inputs(
        cluster_sizes, cluster_noncanonical_counts, bucket_target_fractions, cluster_active_counts
    )

    clusters = list(cluster_sizes.keys())
    sizes = np.array([cluster_sizes[c] for c in clusters], dtype=float)
    ncs = np.array([cluster_noncanonical_counts[c] for c in clusters], dtype=float)
    use_active = cluster_active_counts is not None
    acts = (
        np.array([cluster_active_counts[c] for c in clusters], dtype=float)
        if use_active else np.zeros(len(clusters))
    )
    n_clusters = len(clusters)

    bucket_names = list(bucket_target_fractions.keys())
    n_buckets = len(bucket_names)
    target_fraction = np.array([bucket_target_fractions[b] for b in bucket_names])

    total_size = sizes.sum()
    dataset_nc_rate = ncs.sum() / total_size if total_size else 0.0
    dataset_active_rate = acts.sum() / total_size if total_size else 0.0
    target_size = target_fraction * total_size

    rng = np.random.default_rng(seed)

    # Stage 1: largest-first, most-underfilled-relative-to-target placement.
    order = np.argsort(-sizes)
    assign = np.zeros(n_clusters, dtype=int)
    bsize = np.zeros(n_buckets)
    bnc = np.zeros(n_buckets)
    bact = np.zeros(n_buckets)
    for idx in order:
        b = int(np.argmin(bsize - target_size))
        assign[idx] = b
        bsize[b] += sizes[idx]
        bnc[b] += ncs[idx]
        bact[b] += acts[idx]

    def bucket_term(b: int, size: float, nc: float, act: float) -> float:
        term = ((size - target_size[b]) / target_size[b]) ** 2
        if size > 0:
            term += lambda_noncanonical * (nc / size - dataset_nc_rate) ** 2
            if use_active:
                term += lambda_active * (act / size - dataset_active_rate) ** 2
        return float(term)

    # Stage 2: randomized swap local search; a swap only changes two buckets,
    # so only those two terms are recomputed.
    if n_clusters >= 2:
        for _ in range(n_iterations):
            i, j = rng.integers(0, n_clusters, size=2)
            a, b = assign[i], assign[j]
            if a == b:
                continue
            d_size, d_nc, d_act = sizes[j] - sizes[i], ncs[j] - ncs[i], acts[j] - acts[i]
            old = bucket_term(a, bsize[a], bnc[a], bact[a]) + bucket_term(
                b, bsize[b], bnc[b], bact[b]
            )
            new = bucket_term(a, bsize[a] + d_size, bnc[a] + d_nc, bact[a] + d_act) + bucket_term(
                b, bsize[b] - d_size, bnc[b] - d_nc, bact[b] - d_act
            )
            if new < old:
                assign[i], assign[j] = b, a
                bsize[a] += d_size; bnc[a] += d_nc; bact[a] += d_act
                bsize[b] -= d_size; bnc[b] -= d_nc; bact[b] -= d_act

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
    row_counts: list[int] | None = None,
    active_counts: list[int] | None = None,
    test_size: float = 0.2,
    n_folds: int = 5,
    lambda_noncanonical: float = 8.0,
    lambda_active: float = 0.0,
    n_iterations: int = 40_000,
    seed: int = 42,
) -> SplitAssignment:
    """Carves whole clusters into a "test" bucket (target `test_size`) and
    `n_folds` CV folds (target `(1 - test_size) / n_folds` each), balanced
    toward the dataset-wide has_noncanonical rate computed from
    `has_noncanonical` itself (never hardcoded).

    `row_counts[i]` = number of (peptide, organism) records of peptide i and
    `active_counts[i]` = how many of them are active. When given, bucket
    size / non-canonical rate / active rate are all measured over records
    (what a model is scored on) instead of unique peptides; without them
    every peptide weighs 1 and no active term is applied. Peptides are still
    moved only as part of whole clusters.

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
    if row_counts is None:
        row_counts = [1] * len(peptide_ids)
        if active_counts is not None:
            raise BucketingError("active_counts requires row_counts")
    if len(row_counts) != len(peptide_ids):
        raise BucketingError("row_counts must be the same length as peptide_ids")
    if active_counts is not None:
        if len(active_counts) != len(peptide_ids):
            raise BucketingError("active_counts must be the same length as peptide_ids")
        if any(a > r for a, r in zip(active_counts, row_counts)):
            raise BucketingError("active_counts cannot exceed row_counts")
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
    cluster_active: dict[int, int] = {}
    for k, (community, is_nc) in enumerate(zip(communities, has_noncanonical)):
        cluster_sizes[community] = cluster_sizes.get(community, 0) + row_counts[k]
        cluster_nc[community] = cluster_nc.get(community, 0) + (row_counts[k] if is_nc else 0)
        cluster_active[community] = cluster_active.get(community, 0) + (
            active_counts[k] if active_counts is not None else 0
        )

    fold_names = [f"fold_{i}" for i in range(n_folds)]
    fold_fraction = (1.0 - test_size) / n_folds
    bucket_target_fractions = {"test": test_size, **{name: fold_fraction for name in fold_names}}

    cluster_bucket = assign_clusters_to_buckets(
        cluster_sizes, cluster_nc, bucket_target_fractions,
        cluster_active_counts=cluster_active if active_counts is not None else None,
        lambda_noncanonical=lambda_noncanonical, lambda_active=lambda_active,
        n_iterations=n_iterations, seed=seed,
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
        members = [k for k, c in enumerate(communities) if cluster_bucket[c] == name]
        n_rows = sum(row_counts[k] for k in members)
        n_nc_peptides = sum(1 for k in members if has_noncanonical[k])
        n_nc_rows = sum(row_counts[k] for k in members if has_noncanonical[k])
        stats = {
            "n_peptides": len(members),
            "n_rows": n_rows,
            "target_size_fraction": target_fraction,
            "has_noncanonical_fraction": (n_nc_peptides / len(members)) if members else 0.0,
            "has_noncanonical_fraction_rows": (n_nc_rows / n_rows) if n_rows else 0.0,
        }
        if active_counts is not None:
            stats["active_fraction_rows"] = (
                sum(active_counts[k] for k in members) / n_rows if n_rows else 0.0
            )
        bucket_stats[name] = stats

    return SplitAssignment(
        split_by_peptide_id=split_by_peptide_id,
        fold_by_peptide_id=fold_by_peptide_id,
        community_by_peptide_id=community_by_peptide_id,
        bucket_stats=bucket_stats,
    )
