"""Tree-model counterpart of ``soamp.engine.cross_validation.train_and_evaluate_fold``.

Same contract on purpose, so ``pipeline/train_cv.py`` (the single CV entrypoint)
and the W&B schema (``cv_results``, ``cv_loss_curves``, ``cv_snapshots``,
``cv_val_predictions.csv.gz``) are identical for every architecture family and
the analysis step needs no per-architecture code:

  * same featurized inputs (scaled peptide vector from ``build_dataset``; an index
    organism featurizer becomes a one-hot block, a vector one is used as-is);
  * same class balancing (``pos_weight`` -> sample weight on positives) and the
    same *weighted* BCE as the torch loss, so ``loss`` is comparable to the MLPs';
  * a boosting round plays the role of an epoch: ``on_epoch_end`` fires once per
    round and snapshots fire after rounds N, 2N, ... and the last one.

Model: scikit-learn's HistGradientBoostingClassifier (histogram gradient boosting,
the LightGBM/XGBoost algorithm family). It is already a pinned dependency, needs no
system OpenMP library and gives per-round scores via ``staged_decision_function``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from soamp.data.factory import build_dataset
from soamp.data.torch_dataset import LABEL_TO_INT
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.metrics import compute_binary_metrics, logits_to_predictions


class TreeModelError(ValueError):
    """Raised for inputs a tree model cannot use (e.g. graph peptide input)."""


def weighted_bce_from_logits(logits: np.ndarray, labels: np.ndarray, pos_weight: float | None) -> float:
    """Mean of torch's BCEWithLogitsLoss(pos_weight=...) -- numerically stable."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    pw = 1.0 if pos_weight is None else float(pos_weight)
    return float(np.mean(-(pw * y * -np.logaddexp(0.0, -z) + (1.0 - y) * -np.logaddexp(0.0, z))))


def dataset_to_matrix(dataset, *, organism_vocab_size: int | None, organism_output_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) from a PeptideOrganismDataset: scaled peptide vector ++ organism block."""
    if dataset.peptide_graphs is not None:
        raise TreeModelError("tree models need a vector peptide representation, not molecular_graph")
    peptide, organism, labels = [], [], []
    for i in range(len(dataset)):
        p, o, y = dataset[i]
        peptide.append(p.numpy())
        organism.append(o.numpy())
        labels.append(int(y))
    peptide_x = np.vstack(peptide).astype(np.float32)
    if organism_output_kind == "index":
        idx = np.asarray(organism, dtype=int).reshape(-1)
        organism_x = np.zeros((len(idx), int(organism_vocab_size)), dtype=np.float32)
        organism_x[np.arange(len(idx)), idx] = 1.0
    else:
        organism_x = np.vstack(organism).astype(np.float32)
    return np.hstack([peptide_x, organism_x]), np.asarray(labels, dtype=int)


def count_tree_parameters(params: dict) -> int:
    """Upper bound on leaves (rounds x max leaves) -- the tree analogue of n_parameters."""
    return int(params["max_iter"]) * int(params["max_leaf_nodes"])


def train_and_evaluate_fold_tree(
    row_groups: dict[str, list[dict]],
    *,
    eval_groups: tuple[str, ...] = ("fit", "val"),
    seed: int,
    peptide_method: str,
    organism_method: str,
    params: dict,
    peptide_method_kwargs: dict | None = None,
    organism_method_kwargs: dict | None = None,
    class_balancing_mode: str = "auto",
    class_balancing_fixed_pos_weight: float | None = None,
    on_epoch_end: Callable[[dict], None] | None = None,
    snapshot_every_n_epochs: int | None = None,
    on_snapshot: Callable[[dict], None] | None = None,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """One fold, same return as the torch version: (metrics_by_group, results_df)."""
    bundle = build_dataset(
        row_groups=row_groups, peptide_method=peptide_method, peptide_method_kwargs=peptide_method_kwargs,
        organism_method=organism_method, organism_method_kwargs=organism_method_kwargs,
    )
    feat = bundle.featurization
    mats = {
        g: dataset_to_matrix(ds, organism_vocab_size=feat.organism_vocab_size, organism_output_kind=feat.organism_output_kind)
        for g, ds in bundle.datasets.items()
    }
    x_fit, y_fit = mats["fit"]
    pos_weight = resolve_pos_weight(class_balancing_mode, class_balancing_fixed_pos_weight, [LABEL_TO_INT[r["label"]] for r in row_groups["fit"]])
    sample_weight = np.where(y_fit == 1, pos_weight, 1.0) if pos_weight is not None else None

    model = HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)
    model.fit(x_fit, y_fit, sample_weight=sample_weight)
    n_rounds = int(model.n_iter_)

    snapshot_enabled = snapshot_every_n_epochs is not None and on_snapshot is not None
    if snapshot_enabled:
        if snapshot_every_n_epochs < 1:
            raise ValueError(f"snapshot_every_n_epochs must be >= 1, got {snapshot_every_n_epochs}")
        if "fit" not in mats or "val" not in mats:
            raise ValueError("snapshots need both a 'fit' and a 'val' group in row_groups")
    if on_epoch_end is not None and "val" not in mats:
        raise ValueError("on_epoch_end needs a 'val' group in row_groups")

    if on_epoch_end is not None or snapshot_enabled:
        x_val, y_val = mats["val"]
        # staged_decision_function yields the raw log-odds after each boosting round
        for k, (fit_z, val_z) in enumerate(zip(model.staged_decision_function(x_fit), model.staged_decision_function(x_val)), start=1):
            if on_epoch_end is not None:
                vm = compute_binary_metrics(val_z, y_val)
                on_epoch_end({
                    "epoch": k, "fit_loss": weighted_bce_from_logits(fit_z, y_fit, pos_weight),
                    "val_loss": weighted_bce_from_logits(val_z, y_val, pos_weight),
                    "val_auroc": vm["auroc"], "val_mcc": vm["mcc"], "val_f1": vm["f1"],
                })
            if snapshot_enabled and (k % snapshot_every_n_epochs == 0 or k == n_rounds):
                on_snapshot({
                    "epoch": k,
                    "metrics": {
                        "fit": {**compute_binary_metrics(fit_z, y_fit), "loss": weighted_bce_from_logits(fit_z, y_fit, pos_weight)},
                        "val": {**compute_binary_metrics(val_z, y_val), "loss": weighted_bce_from_logits(val_z, y_val, pos_weight)},
                    },
                    "val_predictions": pd.DataFrame({
                        "peptide_id": [r["peptide_id"] for r in row_groups["val"]],
                        "organism": [r["organism"] for r in row_groups["val"]],
                        "has_noncanonical": [r["has_noncanonical"] for r in row_groups["val"]],
                        "true_label_int": y_val.astype(int),
                        "logit": np.asarray(val_z, dtype=float),
                    }),
                })

    metrics_by_group, frames = {}, []
    for group in eval_groups:
        x, y = mats[group]
        logits = np.asarray(model.decision_function(x), dtype=float)
        metrics_by_group[group] = compute_binary_metrics(logits, y)
        df = pd.DataFrame(row_groups[group]).reset_index(drop=True)
        df["true_label_int"] = y.astype(int)
        df["pred_label_int"] = logits_to_predictions(logits)
        df["logit"] = logits
        df["prob"] = 1.0 / (1.0 + np.exp(-logits))
        df["correct"] = df["true_label_int"] == df["pred_label_int"]
        df["eval_group"] = group
        frames.append(df)
    return metrics_by_group, pd.concat(frames, ignore_index=True)
