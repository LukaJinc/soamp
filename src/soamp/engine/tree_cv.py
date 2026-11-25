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

Two model families share this path (``TREE_ARCHITECTURES``):

  * ``hist_gradient_boosting_classifier`` -- scikit-learn's HistGradientBoostingClassifier:
    a pinned dependency that needs no system OpenMP library and gives per-round scores
    via ``staged_decision_function``;
  * ``xgboost_classifier`` -- XGBoost (``tree_method="hist"``, CPU), fitted and scored in an
    isolated worker process (``xgb_worker.py``): its OpenMP runtime collides with torch's and
    scikit-learn's in one process on macOS (segfault / hang), and this way this process never
    imports xgboost. Per-round raw margins are log-odds, base score included.

Each family is wrapped in a tiny adapter (``fit_tree`` -> object with ``n_rounds``,
``staged_logits(name)`` and ``logits(name)`` for the matrices handed to it); everything after the
adapter is shared.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from soamp.data.factory import build_dataset
from soamp.data.torch_dataset import LABEL_TO_INT
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.metrics import compute_binary_metrics, logits_to_predictions


HGB = "hist_gradient_boosting_classifier"
XGB = "xgboost_classifier"
LOGREG = "logistic_regression_classifier"   # linear baseline: one fit = one "round"
_XGB_WORKER = Path(__file__).with_name("xgb_worker.py")   # see that file for why XGBoost runs in its own process


class TreeModelError(ValueError):
    """Raised for inputs a tree model cannot use (e.g. graph peptide input)."""


class _FittedTree:
    """A fitted boosting model together with the matrices it will be asked to score, by name."""

    n_rounds: int

    def staged_logits(self, name: str):
        """Yield the raw log-odds of set ``name`` after round 1, 2, ..., n_rounds."""
        raise NotImplementedError

    def logits(self, name: str) -> np.ndarray:
        raise NotImplementedError


class _FittedHGB(_FittedTree):
    def __init__(self, model: HistGradientBoostingClassifier, sets: dict[str, np.ndarray]):
        self.model, self.sets, self.n_rounds = model, sets, int(model.n_iter_)

    def staged_logits(self, name):
        return self.model.staged_decision_function(self.sets[name])

    def logits(self, name):
        return np.asarray(self.model.decision_function(self.sets[name]), dtype=float)


class _FittedLogReg(_FittedTree):
    """A single fit, exposed as one round so the curves/snapshots code needs no special case."""

    def __init__(self, model: LogisticRegression, sets: dict[str, np.ndarray]):
        self.model, self.sets, self.n_rounds = model, sets, 1

    def staged_logits(self, name):
        yield self.logits(name)

    def logits(self, name):
        return np.asarray(self.model.decision_function(self.sets[name]), dtype=float)


class _FittedXGB(_FittedTree):
    """Holds what the isolated worker computed (the model itself never enters this process)."""

    def __init__(self, out: dict[str, np.ndarray]):
        self.out, self.n_rounds = out, int(out["n_rounds"])

    def staged_logits(self, name):
        return iter(self.out[f"staged_{name}"])

    def logits(self, name):
        return self.out[f"final_{name}"]


def _run_xgb_worker(params: dict, seed: int, x: np.ndarray, y: np.ndarray, w, sets: dict[str, np.ndarray]) -> dict:
    with tempfile.TemporaryDirectory(prefix="soamp_xgb_") as tmp:
        d = Path(tmp)
        (d / "job.json").write_text(json.dumps({"params": params, "seed": int(seed)}))
        np.savez(d / "train.npz", x=x, y=y, **({} if w is None else {"w": np.asarray(w, dtype=float)}))
        np.savez(d / "sets.npz", **sets)
        proc = subprocess.run([sys.executable, str(_XGB_WORKER), str(d)], capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            if any("No module named 'xgboost'" in line for line in tail):
                raise TreeModelError(f"architecture {XGB!r} needs the xgboost package (on macOS also libomp): {tail[-1]}")
            raise TreeModelError(f"xgboost worker failed (exit {proc.returncode}): " + " | ".join(tail))
        with np.load(d / "out.npz") as out:
            return {k: out[k] for k in out.files}   # read fully: the temp dir disappears on exit


def fit_tree(architecture: str, params: dict, x: np.ndarray, y: np.ndarray, sample_weight, seed: int,
             sets: dict[str, np.ndarray]) -> _FittedTree:
    """Fit one boosting model of the requested family (no early stopping: every round counts) and
    make ``sets`` (name -> matrix) scoreable by name."""
    if architecture == HGB:
        model = HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)
        model.fit(x, y, sample_weight=sample_weight)
        return _FittedHGB(model, sets)
    if architecture == XGB:
        return _FittedXGB(_run_xgb_worker(params, seed, x, y, sample_weight, sets))
    if architecture == LOGREG:
        model = LogisticRegression(C=params["C"], max_iter=params["max_iter"], solver="lbfgs", random_state=seed)
        model.fit(x, y, sample_weight=sample_weight)
        return _FittedLogReg(model, sets)
    raise TreeModelError(f"unknown tree architecture {architecture!r}; known: {[HGB, XGB, LOGREG]}")


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


def count_tree_parameters(params: dict, architecture: str = HGB, n_features: int | None = None) -> int:
    """Upper bound on leaves (rounds x max leaves per tree) -- the tree analogue of n_parameters.
    HGB: ``max_iter * max_leaf_nodes``; XGBoost: ``n_estimators * 2**max_depth``;
    logistic regression: weights + bias (``n_features + 1``, 0 if the width is unknown)."""
    if architecture == LOGREG:
        return int(n_features) + 1 if n_features is not None else 0
    if architecture == XGB:
        return int(params["n_estimators"]) * 2 ** int(params["max_depth"])
    return int(params["max_iter"]) * int(params["max_leaf_nodes"])


def train_and_evaluate_fold_tree(
    row_groups: dict[str, list[dict]],
    *,
    eval_groups: tuple[str, ...] = ("fit", "val"),
    seed: int,
    peptide_method: str,
    organism_method: str,
    params: dict,
    architecture: str = HGB,
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

    model = fit_tree(architecture, params, x_fit, y_fit, sample_weight, seed, {g: m[0] for g, m in mats.items()})
    n_rounds = model.n_rounds

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
        # the raw log-odds after each boosting round (same unit for both families)
        for k, (fit_z, val_z) in enumerate(zip(model.staged_logits("fit"), model.staged_logits("val")), start=1):
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
        logits = model.logits(group)
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
