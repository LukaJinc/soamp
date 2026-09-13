"""K-fold cross-validation over the Leiden-community folds in
data/train_folds_leiden.csv -- lifted out of scripts/EDA/build_dataset_build_model.ipynb
(and its kfold_demo sibling) into tested, canonical library code, per the
same "two cooks" reasoning as soamp.data.splitting's val-split extraction:
the fold-evaluation loop existed only as copy-pasted notebook cells, and the
two copies had already drifted -- one joins the fold file on the correct key
(`sequence`), the other on an unrelated ID space (`node_id` vs `peptide_id`)
that only coincidentally overlaps in range, silently misassigning folds.

No wandb/tracker coupling here (per CLAUDE.md sec 6) -- pipeline/train_cv.py
owns the wandb.init/log/Artifact calls; everything in this module is plain
data in, data out, so it's testable without mocking a tracker.
"""
import csv
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from soamp.data.factory import build_dataset
from soamp.data.torch_dataset import LABEL_TO_INT
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.metrics import compute_binary_metrics, logits_to_predictions
from soamp.engine.tracking import watch_model
from soamp.engine.trainer import Trainer
from soamp.model.factory import build_model
from soamp.utils.device import resolve_device


class CrossValidationError(ValueError):
    """Raised when a row's sequence has no fold assignment."""


def load_fold_assignments(folds_csv_path: str | Path) -> dict[str, str]:
    """Reads train_folds_leiden.csv, keyed by **sequence** -- the verified
    join key (train_folds_leiden.csv's `node_id` is an igraph vertex index
    assigned when the fold-generation notebook built its similarity graph;
    it is not this dataset's `peptide_id`, and joining the two directly
    silently misassigns folds since they're unrelated ID spaces that happen
    to overlap in range). Values are the `fold_id` field as written in the
    CSV (a string, e.g. "0".."4")."""
    with open(folds_csv_path, newline="") as f:
        return {row["sequence"]: row["fold_id"] for row in csv.DictReader(f)}


def assign_fold_ids(rows: list[dict], fold_by_sequence: dict[str, str]) -> list[dict]:
    """Returns new row dicts with a `fold_id` key added, looked up by each
    row's `sequence`. Raises CrossValidationError listing every row with no
    fold assignment, rather than silently dropping or misassigning it."""
    missing = sorted({r["sequence"] for r in rows if r["sequence"] not in fold_by_sequence})
    if missing:
        raise CrossValidationError(
            f"{len(missing)} sequence(s) have no fold assignment in the folds CSV: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return [{**row, "fold_id": fold_by_sequence[row["sequence"]]} for row in rows]


def train_and_evaluate_fold(
    row_groups: dict[str, list[dict]],
    *,
    eval_groups: tuple[str, ...] = ("fit", "val"),
    epochs: int,
    seed: int,
    peptide_method: str,
    organism_method: str,
    architecture: str,
    architecture_kwargs: dict | None = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    class_balancing_mode: str = "auto",
    class_balancing_fixed_pos_weight: float | None = None,
    device: "torch.device | str | None" = None,
    watch: bool = False,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """One fold: build_dataset(row_groups=...) -> build_model -> train for
    `epochs` -> evaluate on every eval_groups member. Reproduces
    scripts/EDA/build_dataset_build_model.ipynb's `train_and_evaluate` body
    exactly, generalized to take class-balancing and device as parameters
    (the notebook hardcodes "auto"/None and is CPU-only) -- defaults
    reproduce the notebook's behavior unchanged.

    Returns (metrics_by_group, results_df): metrics_by_group maps each
    eval_groups member to a compute_binary_metrics() dict; results_df is
    row-level -- every row_groups[group] dict's own columns plus
    true_label_int/pred_label_int/logit/prob/correct/eval_group, concatenated
    across eval_groups (one row per (input row, eval_group) pair, since a
    "fit" row is also scored as part of the "fit" eval_group).
    """
    torch.manual_seed(seed)
    if device is None:
        resolved_device = resolve_device("auto")
    elif isinstance(device, torch.device):
        resolved_device = device
    else:
        resolved_device = resolve_device(device)

    bundle = build_dataset(
        row_groups=row_groups, peptide_method=peptide_method, organism_method=organism_method
    )
    model = build_model(bundle, architecture=architecture, **(architecture_kwargs or {}))

    fit_loader = DataLoader(bundle.datasets["fit"], batch_size=batch_size, shuffle=True)

    fit_labels = [LABEL_TO_INT[row["label"]] for row in row_groups["fit"]]
    pos_weight = resolve_pos_weight(class_balancing_mode, class_balancing_fixed_pos_weight, fit_labels)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=resolved_device) if pos_weight is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, loss_fn, device=resolved_device)

    if watch:
        # Watches the real model about to be trained -- gradient/parameter
        # histograms and the computation graph populate from its actual
        # forward/backward passes below, not a throwaway preview model.
        watch_model(model)

    for _ in range(epochs):
        trainer.train_epoch(fit_loader)

    # Each eval_loader has shuffle=False, so its eval_out arrays line up 1:1
    # with row_groups[group] in order -- safe to reconstruct a per-row table.
    metrics_by_group = {}
    group_frames = []
    for group in eval_groups:
        eval_loader = DataLoader(bundle.datasets[group], batch_size=batch_size)
        eval_out = trainer.evaluate(eval_loader)
        metrics_by_group[group] = compute_binary_metrics(eval_out["logits"], eval_out["labels"])

        probs = 1.0 / (1.0 + np.exp(-eval_out["logits"]))
        group_df = pd.DataFrame(row_groups[group]).reset_index(drop=True)
        group_df["true_label_int"] = eval_out["labels"].astype(int)
        group_df["pred_label_int"] = logits_to_predictions(eval_out["logits"])
        group_df["logit"] = eval_out["logits"]
        group_df["prob"] = probs
        group_df["correct"] = group_df["true_label_int"] == group_df["pred_label_int"]
        group_df["eval_group"] = group
        group_frames.append(group_df)

    results_df = pd.concat(group_frames, ignore_index=True)
    return metrics_by_group, results_df


def summarize_cv_metrics(fold_metrics_df: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    """Cross-fold mean/std per eval_group, for the given metric columns --
    generalizes the notebooks' hardcoded ["accuracy", "f1", "auroc"] groupby
    to an arbitrary metric list (now including precision/recall). Returns a
    DataFrame indexed by eval_group with a (metric, agg) MultiIndex on
    columns, exactly as fold_metrics_df.groupby("eval_group")[metric_names]
    .agg(["mean", "std"]) would -- callers read
    summary_df.loc[eval_group, (metric, "mean" | "std")]."""
    return fold_metrics_df.groupby("eval_group")[metric_names].agg(["mean", "std"])
