"""Test + 5-fold split of the *labelled* dev dataset, from precomputed clusters (pure logic).

The clusters (alignment-identity clusters for linear peptides, Murcko ring-system clusters
for non-linear ones) are built once, on every peptide of the target strains, by
scripts/EDA/09_data_processing.ipynb. That clustering is kept as it is on purpose: two
labelled peptides can be homologous only through a peptide that has no labelled pair, and
re-clustering the labelled subset would lose that link.

What is re-derived here is the *assignment of whole clusters to buckets*, with the same
stratified swap search as every other split in the repo
(``soamp.splitting.bucketing.stratified_cluster_split``), but weighted by what a model is
trained and scored on: the labelled (peptide, organism) pairs. Balanced: bucket size,
non-linear share and active share. Peptides without a labelled pair carry no weight and
are not part of the result.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from soamp.splitting.bucketing import PeptideGroup, stratified_cluster_split
from soamp.splitting.config import BucketingConfig


class FinalSplitError(ValueError):
    pass


@dataclass(frozen=True)
class FinalSplit:
    table: pd.DataFrame     # peptide_id, cluster_id, cluster_kind, split, fold_id (NaN for test)
    summary: pd.DataFrame   # one row per bucket
    sidecar: dict           # parameters and counts, for reproducibility
    organism_tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # "<view>_<rows|active_%>" -> organism x bucket


def bucket_of(table: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(table["split"] == "test", "test", "fold_" + table["fold_id"].astype("Int64").astype(str)),
                     index=table.index)


def _organism_groups(pairs: pd.DataFrame, peptide_index: pd.Index, view: str, weight: float) -> list[PeptideGroup]:
    """One group per organism: per-peptide labelled pairs and active pairs (``view`` only names the group)."""
    groups = []
    for organism in sorted(pairs["organism"].unique()):
        sub = pairs[pairs["organism"] == organism]
        per = sub.assign(_active=sub["label"].eq("active")).groupby("peptide_id").agg(
            rows=("label", "size"), active=("_active", "sum")).reindex(peptide_index, fill_value=0)
        groups.append(PeptideGroup(f"{view}:{organism}", weight, per["rows"].astype(int).tolist(),
                                   per["active"].astype(int).tolist()))
    return groups


def split_labelled_peptides(pairs: pd.DataFrame, clusters: pd.DataFrame, cfg: BucketingConfig) -> FinalSplit:
    """``pairs``: one row per labelled (peptide, organism) with ``peptide_id``, ``label``
    ('active'/'inactive') and ``has_noncanonical``. ``clusters``: ``peptide_id``, ``cluster_id``,
    ``cluster_kind`` for (at least) every peptide in ``pairs``. With ``organism`` in ``pairs`` and
    ``cfg.lambda_organism_active*`` > 0 the per-organism active rate is balanced as well, over all pairs
    and/or over the linear pairs only."""
    per_peptide = (pairs.assign(_active=pairs["label"].eq("active"))
                        .groupby("peptide_id")
                        .agg(rows=("label", "size"), active=("_active", "sum"), nc=("has_noncanonical", "first"))
                        .sort_index())
    clusters = clusters.drop_duplicates("peptide_id").set_index("peptide_id")
    missing = sorted(set(per_peptide.index) - set(clusters.index))
    if missing:
        raise FinalSplitError(f"{len(missing)} labelled peptides have no cluster, e.g. {missing[:5]}")
    cl = clusters.loc[per_peptide.index]

    peptide_ids = per_peptide.index.tolist()
    groups: list[PeptideGroup] = []
    if cfg.lambda_organism_active > 0 or cfg.lambda_organism_active_linear > 0:
        if "organism" not in pairs.columns:
            raise FinalSplitError("per-organism balancing needs an 'organism' column in pairs")
        if cfg.lambda_organism_active > 0:
            groups += _organism_groups(pairs, per_peptide.index, "all", cfg.lambda_organism_active)
        if cfg.lambda_organism_active_linear > 0:
            linear_pairs = pairs[~pairs["has_noncanonical"].astype(bool)]
            groups += _organism_groups(linear_pairs, per_peptide.index, "linear", cfg.lambda_organism_active_linear)
    node_communities = pd.DataFrame({"node_id": range(len(peptide_ids)), "community": cl["cluster_id"].to_numpy()})
    assignment = stratified_cluster_split(
        node_communities, peptide_ids, per_peptide["nc"].astype(bool).tolist(),
        row_counts=per_peptide["rows"].astype(int).tolist(),
        active_counts=per_peptide["active"].astype(int).tolist(),
        groups=groups,
        test_size=cfg.test_size, n_folds=cfg.n_folds,
        lambda_noncanonical=cfg.lambda_noncanonical, lambda_active=cfg.lambda_active,
        n_iterations=cfg.n_iterations, seed=cfg.seed,
    )
    table = pd.DataFrame({
        "peptide_id": peptide_ids,
        "cluster_id": cl["cluster_id"].to_numpy(),
        "cluster_kind": cl["cluster_kind"].to_numpy(),
        "split": [assignment.split_by_peptide_id[p] for p in peptide_ids],
        "fold_id": [np.nan if assignment.fold_by_peptide_id[p] is None else float(assignment.fold_by_peptide_id[p])
                    for p in peptide_ids],
    })
    summary = summarise(table, per_peptide, cfg)
    organism_tables = organism_balance_tables(pairs, table) if "organism" in pairs.columns else {}
    sidecar = {
        "method": "stratified_cluster_split over labelled pairs of precomputed clusters",
        "parameters": cfg.model_dump(),
        "labelled_peptides": len(peptide_ids),
        "labelled_pairs": int(per_peptide["rows"].sum()),
        "clusters_used": int(table["cluster_id"].nunique()),
        "cluster_kinds": table.drop_duplicates("cluster_id")["cluster_kind"].value_counts().to_dict(),
        "buckets": summary.reset_index().to_dict("records"),
        "organism_balance": {k: v.reset_index().to_dict("records") for k, v in organism_tables.items()},
    }
    return FinalSplit(table, summary, sidecar, organism_tables)


def organism_balance_tables(pairs: pd.DataFrame, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per organism x bucket pair counts and active % for the full set and the linear subset, with the
    organism's overall value in column ``all`` and the worst bucket deviation in ``max_dev``."""
    bucket = bucket_of(table).set_axis(table["peptide_id"])
    out: dict[str, pd.DataFrame] = {}
    for view, sub in (("full", pairs), ("linear", pairs[~pairs["has_noncanonical"].astype(bool)])):
        d = sub.assign(bucket=sub["peptide_id"].map(bucket), active=sub["label"].eq("active"))
        rows = d.pivot_table(index="organism", columns="bucket", values="active", aggfunc="size", fill_value=0)
        act = (d.pivot_table(index="organism", columns="bucket", values="active", aggfunc="mean") * 100).round(1)
        cols = ["test"] + [c for c in sorted(act.columns) if c != "test"]
        overall = (d.groupby("organism")["active"].mean() * 100).round(1)
        act = act[cols].assign(all=overall)
        act["max_dev"] = act[cols].sub(act["all"], axis=0).abs().max(axis=1).round(1)
        out[f"{view}_rows"] = rows[cols].assign(all=rows.sum(axis=1))
        out[f"{view}_active_%"] = act
    return out


def summarise(table: pd.DataFrame, per_peptide: pd.DataFrame, cfg: BucketingConfig) -> pd.DataFrame:
    t = table.set_index("peptide_id").join(per_peptide)
    t["bucket"] = bucket_of(t.reset_index(drop=True)).to_numpy()
    g = t.groupby("bucket")
    out = pd.DataFrame({
        "peptides": g.size(),
        "clusters": g["cluster_id"].nunique(),
        "pairs": g["rows"].sum(),
        "nonlinear_pairs": g.apply(lambda x: x.loc[x["nc"].astype(bool), "rows"].sum(), include_groups=False),
        "active_pairs": g["active"].sum(),
    })
    total = out["pairs"].sum()
    out["pair_share_%"] = (100 * out["pairs"] / total).round(2)
    out["target_%"] = [100 * cfg.test_size if b == "test" else 100 * (1 - cfg.test_size) / cfg.n_folds for b in out.index]
    out["nonlinear_%"] = (100 * out["nonlinear_pairs"] / out["pairs"]).round(2)
    out["active_%"] = (100 * out["active_pairs"] / out["pairs"]).round(2)
    return out.loc[["test"] + [f"fold_{i}" for i in range(cfg.n_folds)]]


def check_final_split(split: FinalSplit, tolerance_points: float = 2.0) -> None:
    """Hard checks: nothing unassigned, no cluster across two buckets, bucket sizes near target."""
    t, s = split.table, split.summary
    if t["split"].isna().any():
        raise FinalSplitError("a peptide has no bucket")
    spread = bucket_of(t).groupby(t["cluster_id"]).nunique()
    if (spread > 1).any():
        raise FinalSplitError(f"{int((spread > 1).sum())} clusters span more than one bucket")
    worst = (s["pair_share_%"] - s["target_%"]).abs().max()
    if worst >= tolerance_points:
        raise FinalSplitError(f"a bucket is {worst:.2f} points off its target size")
