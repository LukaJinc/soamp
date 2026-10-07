import numpy as np
import pandas as pd
import pytest

from soamp.splitting.bucketing import (
    BucketingError,
    assign_clusters_to_buckets,
    stratified_cluster_split,
)

_TARGETS = {
    "test": 0.2, "fold_0": 0.16, "fold_1": 0.16,
    "fold_2": 0.16, "fold_3": 0.16, "fold_4": 0.16,
}


def _synthetic_clusters(seed=0, n_clusters=200):
    """Reproduces this project's real cluster-distribution shape: a handful
    of large, mostly-canonical clusters and many small clusters, with
    non-canonical peptides concentrated unevenly across cluster sizes --
    not spread proportionally. This lumpiness is exactly what defeated the
    rejected single-pass greedy design (see bucketing.py's module docstring)."""
    rng = np.random.default_rng(seed)
    sizes, ncs = {}, {}
    for c in range(n_clusters):
        size = int(rng.integers(200, 1200) if c < 5 else rng.integers(1, 40))
        nc_rate = rng.choice([0.0, 0.0, 0.0, 0.3, 0.8, 1.0])
        sizes[c] = size
        ncs[c] = int(round(size * nc_rate))
    return sizes, ncs


def test_assign_covers_every_cluster_exactly_once():
    sizes, ncs = _synthetic_clusters()
    result = assign_clusters_to_buckets(sizes, ncs, _TARGETS, n_iterations=2000)
    assert set(result.keys()) == set(sizes.keys())
    assert set(result.values()) <= set(_TARGETS.keys())


def test_assign_is_deterministic_given_seed():
    sizes, ncs = _synthetic_clusters()
    r1 = assign_clusters_to_buckets(sizes, ncs, _TARGETS, seed=7, n_iterations=2000)
    r2 = assign_clusters_to_buckets(sizes, ncs, _TARGETS, seed=7, n_iterations=2000)
    assert r1 == r2


def test_assign_balances_size_reasonably():
    sizes, ncs = _synthetic_clusters()
    result = assign_clusters_to_buckets(sizes, ncs, _TARGETS, n_iterations=5000)
    total = sum(sizes.values())
    bucket_size = {b: 0 for b in _TARGETS}
    for c, b in result.items():
        bucket_size[b] += sizes[c]
    for name, frac in _TARGETS.items():
        target = frac * total
        assert bucket_size[name] == pytest.approx(target, rel=0.25)


def test_assign_stratifies_noncanonical_rate_better_than_unweighted():
    sizes, ncs = _synthetic_clusters()

    def spread(lam):
        result = assign_clusters_to_buckets(
            sizes, ncs, _TARGETS, lambda_noncanonical=lam, n_iterations=5000, seed=1
        )
        bucket_size = {b: 0 for b in _TARGETS}
        bucket_nc = {b: 0 for b in _TARGETS}
        for c, b in result.items():
            bucket_size[b] += sizes[c]
            bucket_nc[b] += ncs[c]
        rates = [bucket_nc[b] / bucket_size[b] for b in _TARGETS if bucket_size[b] > 0]
        return max(rates) - min(rates)

    assert spread(lam=8.0) < spread(lam=0.0)


def test_assign_raises_on_mismatched_keys():
    with pytest.raises(BucketingError):
        assign_clusters_to_buckets({1: 10}, {2: 1}, {"test": 0.2, "train": 0.8})


def test_assign_raises_when_noncanonical_exceeds_size():
    with pytest.raises(BucketingError):
        assign_clusters_to_buckets({1: 5}, {1: 10}, {"test": 0.2, "train": 0.8})


def test_assign_raises_on_targets_not_summing_to_one():
    with pytest.raises(BucketingError):
        assign_clusters_to_buckets({1: 5, 2: 5}, {1: 0, 2: 0}, {"test": 0.2, "train": 0.5})


def test_stratified_cluster_split_whole_cluster_invariant():
    node_communities = pd.DataFrame({"node_id": [0, 1, 2, 3], "community": [0, 0, 1, 1]})
    peptide_ids = ["a", "b", "c", "d"]
    has_noncanonical = [True, False, False, True]
    result = stratified_cluster_split(
        node_communities, peptide_ids, has_noncanonical,
        test_size=0.5, n_folds=2, n_iterations=1000,
    )
    assert result.split_by_peptide_id["a"] == result.split_by_peptide_id["b"]
    assert result.fold_by_peptide_id["a"] == result.fold_by_peptide_id["b"]
    assert result.split_by_peptide_id["c"] == result.split_by_peptide_id["d"]
    assert result.fold_by_peptide_id["c"] == result.fold_by_peptide_id["d"]


def test_stratified_cluster_split_covers_and_disjoint():
    node_communities = pd.DataFrame({"node_id": [0, 1, 2, 3, 4], "community": [0, 1, 2, 3, 4]})
    peptide_ids = ["a", "b", "c", "d", "e"]
    has_noncanonical = [False, False, True, False, True]
    result = stratified_cluster_split(
        node_communities, peptide_ids, has_noncanonical,
        test_size=0.2, n_folds=4, n_iterations=1000,
    )
    assert set(result.split_by_peptide_id.keys()) == set(peptide_ids)
    for pid in peptide_ids:
        if result.split_by_peptide_id[pid] == "test":
            assert result.fold_by_peptide_id[pid] is None
        else:
            assert result.fold_by_peptide_id[pid] in range(4)


def test_stratified_cluster_split_handles_isolated_nodes_not_in_communities_df():
    # node 2 ("c") has no edges at all -- absent from the Leiden output
    # entirely, must still get assigned (as its own singleton community).
    node_communities = pd.DataFrame({"node_id": [0, 1, 3], "community": [0, 0, 1]})
    peptide_ids = ["a", "b", "c", "d"]
    has_noncanonical = [False, False, False, False]
    result = stratified_cluster_split(
        node_communities, peptide_ids, has_noncanonical,
        test_size=0.25, n_folds=3, n_iterations=500,
    )
    assert "c" in result.split_by_peptide_id
    assert result.community_by_peptide_id["c"] not in (
        result.community_by_peptide_id["a"], result.community_by_peptide_id["d"]
    )


def test_stratified_cluster_split_raises_on_bad_test_size():
    node_communities = pd.DataFrame({"node_id": [0, 1], "community": [0, 1]})
    with pytest.raises(BucketingError):
        stratified_cluster_split(node_communities, ["a", "b"], [False, False], test_size=1.5)


def test_stratified_cluster_split_raises_on_bad_n_folds():
    node_communities = pd.DataFrame({"node_id": [0, 1], "community": [0, 1]})
    with pytest.raises(BucketingError):
        stratified_cluster_split(node_communities, ["a", "b"], [False, False], n_folds=1)


def _synthetic_active_counts(sizes, seed=3):
    rng = np.random.default_rng(seed)
    return {c: int(round(s * rng.choice([0.2, 0.6, 0.95, 1.0]))) for c, s in sizes.items()}


def test_assign_balances_active_rate_better_than_ignoring_it():
    sizes, ncs = _synthetic_clusters()
    acts = _synthetic_active_counts(sizes)

    def spread(lam):
        result = assign_clusters_to_buckets(
            sizes, ncs, _TARGETS, cluster_active_counts=acts,
            lambda_active=lam, n_iterations=5000, seed=1,
        )
        size = {b: 0 for b in _TARGETS}
        act = {b: 0 for b in _TARGETS}
        for c, b in result.items():
            size[b] += sizes[c]
            act[b] += acts[c]
        rates = [act[b] / size[b] for b in _TARGETS if size[b] > 0]
        return max(rates) - min(rates)

    assert spread(lam=10.0) < spread(lam=0.0)


def test_assign_raises_when_active_exceeds_size_or_keys_mismatch():
    with pytest.raises(BucketingError):
        assign_clusters_to_buckets(
            {1: 5, 2: 5}, {1: 0, 2: 0}, {"test": 0.5, "train": 0.5},
            cluster_active_counts={1: 6, 2: 0},
        )
    with pytest.raises(BucketingError):
        assign_clusters_to_buckets(
            {1: 5, 2: 5}, {1: 0, 2: 0}, {"test": 0.5, "train": 0.5},
            cluster_active_counts={1: 1},
        )


def test_stratified_split_record_weighted_never_splits_a_cluster_and_reports_rates():
    # 40 two-peptide clusters; peptides carry different numbers of records.
    n = 80
    node_communities = pd.DataFrame({"node_id": range(n), "community": [i // 2 for i in range(n)]})
    peptide_ids = [str(i) for i in range(n)]
    rng = np.random.default_rng(0)
    row_counts = [int(r) for r in rng.integers(1, 4, size=n)]
    active_counts = [int(rng.integers(0, r + 1)) for r in row_counts]
    result = stratified_cluster_split(
        node_communities, peptide_ids, [i % 5 == 0 for i in range(n)],
        row_counts=row_counts, active_counts=active_counts,
        test_size=0.2, n_folds=4, lambda_active=10.0, n_iterations=3000,
    )
    for c in range(n // 2):
        a, b = peptide_ids[2 * c], peptide_ids[2 * c + 1]
        assert result.split_by_peptide_id[a] == result.split_by_peptide_id[b]
        assert result.fold_by_peptide_id[a] == result.fold_by_peptide_id[b]
    assert sum(s["n_rows"] for s in result.bucket_stats.values()) == sum(row_counts)
    assert all("active_fraction_rows" in s for s in result.bucket_stats.values())


def test_stratified_split_rejects_active_without_rows_or_exceeding_rows():
    nc = pd.DataFrame({"node_id": [0, 1], "community": [0, 1]})
    with pytest.raises(BucketingError):
        stratified_cluster_split(nc, ["a", "b"], [False, False], active_counts=[1, 1])
    with pytest.raises(BucketingError):
        stratified_cluster_split(
            nc, ["a", "b"], [False, False], row_counts=[1, 1], active_counts=[2, 0]
        )
