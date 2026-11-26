import numpy as np
import pandas as pd
import pytest

from soamp.splitting.config import BucketingConfig
from soamp.splitting.dev_final_split import (
    FinalSplit, FinalSplitError, bucket_of, check_final_split, split_labelled_peptides,
)

CFG = BucketingConfig(n_iterations=3000)


def _toy(n_clusters=60, seed=0):
    """Clusters of 1-4 peptides; each peptide has 1-3 labelled pairs; some non-linear; ~65% active."""
    rng = np.random.default_rng(seed)
    pair_rows, cluster_rows, pid = [], [], 0
    for c in range(n_clusters):
        kind = "murcko" if c % 4 == 0 else "alignment"
        for _ in range(int(rng.integers(1, 5))):
            cluster_rows.append({"peptide_id": pid, "cluster_id": c, "cluster_kind": kind})
            for _ in range(int(rng.integers(1, 4))):
                pair_rows.append({"peptide_id": pid, "label": "active" if rng.random() < 0.65 else "inactive",
                                  "has_noncanonical": kind == "murcko"})
            pid += 1
    return pd.DataFrame(pair_rows), pd.DataFrame(cluster_rows)


def test_every_peptide_assigned_and_no_cluster_split():
    pairs, clusters = _toy()
    res = split_labelled_peptides(pairs, clusters, CFG)
    assert set(res.table["peptide_id"]) == set(pairs["peptide_id"])
    assert (bucket_of(res.table).groupby(res.table["cluster_id"]).nunique() == 1).all()
    assert set(res.table["split"]) == {"test", "train"}
    assert res.table.loc[res.table["split"] == "test", "fold_id"].isna().all()
    assert set(res.table.loc[res.table["split"] == "train", "fold_id"]) == {0.0, 1.0, 2.0, 3.0, 4.0}


def test_weights_are_labelled_pairs_not_peptides():
    pairs, clusters = _toy()
    res = split_labelled_peptides(pairs, clusters, CFG)
    assert res.summary["pairs"].sum() == len(pairs)
    assert res.summary["peptides"].sum() == pairs["peptide_id"].nunique()
    assert res.sidecar["labelled_pairs"] == len(pairs)


def test_sizes_close_to_targets():
    pairs, clusters = _toy(200)
    res = split_labelled_peptides(pairs, clusters, BucketingConfig(n_iterations=20000))
    check_final_split(res)   # raises if a bucket is >= 2 points off or a cluster is split
    assert abs(res.summary.loc["test", "pair_share_%"] - 20) < 2


def test_active_term_reduces_active_share_spread():
    pairs, clusters = _toy(200, seed=3)
    with_active = split_labelled_peptides(pairs, clusters, BucketingConfig(n_iterations=20000, lambda_active=30))
    without = split_labelled_peptides(pairs, clusters, BucketingConfig(n_iterations=20000, lambda_active=0))
    spread = lambda r: r.summary["active_%"].max() - r.summary["active_%"].min()   # noqa: E731
    assert spread(with_active) <= spread(without)


def test_deterministic_for_a_seed_and_changes_with_it():
    pairs, clusters = _toy()
    a = split_labelled_peptides(pairs, clusters, CFG).table
    b = split_labelled_peptides(pairs, clusters, CFG).table
    c = split_labelled_peptides(pairs, clusters, BucketingConfig(n_iterations=3000, seed=7)).table
    assert a.equals(b) and not a["split"].equals(c["split"])


def test_peptides_without_a_cluster_raise():
    pairs, clusters = _toy()
    with pytest.raises(FinalSplitError, match="no cluster"):
        split_labelled_peptides(pairs, clusters.iloc[:-3], CFG)


def test_extra_clustered_peptides_without_pairs_are_ignored():
    pairs, clusters = _toy()
    extra = pd.DataFrame({"peptide_id": [10_000, 10_001], "cluster_id": [999, 999], "cluster_kind": "alignment"})
    res = split_labelled_peptides(pairs, pd.concat([clusters, extra]), CFG)
    assert 10_000 not in set(res.table["peptide_id"])


def test_check_detects_a_split_cluster():
    pairs, clusters = _toy()
    res = split_labelled_peptides(pairs, clusters, CFG)
    bad = res.table.copy()
    big = bad["cluster_id"].value_counts().index[0]
    idx = bad.index[bad["cluster_id"] == big]
    bad.loc[idx[0], ["split", "fold_id"]] = ["test", np.nan] if bad.loc[idx[0], "split"] != "test" else ["train", 0.0]
    with pytest.raises(FinalSplitError, match="span more than one bucket"):
        check_final_split(FinalSplit(bad, res.summary, res.sidecar))


def _toy_organisms(n_clusters=120, seed=0):
    """Like ``_toy`` plus an ``organism`` column whose active rate differs per organism and per cluster,
    so balancing only the overall active share leaves the per-organism shares uneven."""
    pairs, clusters = _toy(n_clusters, seed)
    rng = np.random.default_rng(seed + 1)
    cluster_of = clusters.set_index("peptide_id")["cluster_id"]
    cluster_bias = {c: rng.uniform(0.1, 0.9) for c in clusters["cluster_id"].unique()}
    organisms = rng.choice(["org_a", "org_b", "org_c"], size=len(pairs), p=[0.5, 0.3, 0.2])
    pairs = pairs.assign(organism=organisms)
    p_active = np.where(pairs["organism"] == "org_c", pairs["peptide_id"].map(cluster_of).map(cluster_bias), 0.6)
    pairs["label"] = np.where(rng.random(len(pairs)) < p_active, "active", "inactive")
    return pairs, clusters


def _worst_organism_deviation(res, view="full"):
    return res.organism_tables[f"{view}_active_%"]["max_dev"].max()


def test_organism_term_reduces_per_organism_active_spread():
    pairs, clusters = _toy_organisms()
    base = dict(n_iterations=30000, lambda_active=30)
    off = split_labelled_peptides(pairs, clusters, BucketingConfig(**base))
    on = split_labelled_peptides(pairs, clusters, BucketingConfig(**base, lambda_organism_active=30))
    assert _worst_organism_deviation(on) < _worst_organism_deviation(off)
    assert (bucket_of(on.table).groupby(on.table["cluster_id"]).nunique() == 1).all()
    check_final_split(on)


def test_linear_organism_term_balances_the_linear_view():
    pairs, clusters = _toy_organisms(seed=2)
    base = dict(n_iterations=30000, lambda_active=30, lambda_organism_active=5)
    off = split_labelled_peptides(pairs, clusters, BucketingConfig(**base))
    on = split_labelled_peptides(pairs, clusters, BucketingConfig(**base, lambda_organism_active_linear=30))
    assert _worst_organism_deviation(on, "linear") < _worst_organism_deviation(off, "linear")


def test_organism_weights_zero_keep_the_old_assignment_and_tables_still_reported():
    pairs, clusters = _toy_organisms()
    with_col = split_labelled_peptides(pairs, clusters, CFG)
    without_col = split_labelled_peptides(pairs.drop(columns="organism"), clusters, CFG)
    assert with_col.table.equals(without_col.table)
    assert set(with_col.organism_tables) == {"full_rows", "full_active_%", "linear_rows", "linear_active_%"}
    assert without_col.organism_tables == {}


def test_organism_balancing_needs_an_organism_column():
    pairs, clusters = _toy()
    with pytest.raises(FinalSplitError):
        split_labelled_peptides(pairs, clusters, BucketingConfig(n_iterations=100, lambda_organism_active=5))
