"""K-fold cross-validation over the CV folds in data/peptide_split.csv
(soamp.splitting) -- lifted out of scripts/EDA/build_dataset_build_model.ipynb
(and its kfold_demo sibling) into tested, canonical library code, per the
same "two cooks" reasoning as soamp.data.splitting's val-split extraction:
the fold-evaluation loop existed only as copy-pasted notebook cells, and the
two copies had already drifted -- one joined the old fold file on the
correct key, the other on an unrelated ID space (`node_id` vs `peptide_id`)
that only coincidentally overlaps in range, silently misassigning folds.

Folds are keyed by `peptide_id` (peptide_split.csv's own key), never by
`sequence`: sequences containing the 'X' placeholder are shared by different
molecules that may legitimately land in different folds, so a sequence-keyed
lookup would silently collapse them onto one arbitrary fold.

No wandb/tracker coupling here (per CLAUDE.md sec 6) -- pipeline/train_cv.py
owns the wandb.init/log/Artifact calls; everything in this module is plain
data in, data out, so it's testable without mocking a tracker.
"""
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from soamp.data.factory import build_dataset
from soamp.data.loaders import build_loader
from soamp.data.torch_dataset import LABEL_TO_INT
from soamp.engine.class_balancing import resolve_pos_weight
from soamp.engine.metrics import compute_binary_metrics, logits_to_predictions
from soamp.engine.tracking import watch_model
from soamp.engine.trainer import Trainer
from soamp.model.factory import build_model
from soamp.utils.device import resolve_device


class CrossValidationError(ValueError):
    """Raised when a row's sequence has no fold assignment."""


def load_peptide_split(split_csv_path: str | Path) -> dict[str, dict]:
    """Reads peptide_split.csv into {peptide_id: {"split": "train"|"test",
    "fold_id": "0".."4" | None}} (fold_id is None for test peptides)."""
    out: dict[str, dict] = {}
    with open(split_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fold = row["fold_id"]
            out[row["peptide_id"]] = {
                "split": row["split"],
                "fold_id": str(int(float(fold))) if fold not in ("", None) else None,
            }
    return out


def assign_train_folds(rows: list[dict], peptide_split: dict[str, dict]) -> list[dict]:
    """Returns new row dicts for the rows whose peptide is in the `train`
    split of `peptide_split`, each with a `fold_id` key added (looked up by
    `peptide_id`). `test` rows are dropped, never returned. Raises
    CrossValidationError if any row's peptide_id is absent from
    `peptide_split`, or a train peptide has no fold, rather than silently
    dropping or misassigning it."""
    missing = sorted({r["peptide_id"] for r in rows if r["peptide_id"] not in peptide_split})
    if missing:
        raise CrossValidationError(
            f"{len(missing)} peptide_id(s) have no entry in the split file: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    out = []
    for row in rows:
        entry = peptide_split[row["peptide_id"]]
        if entry["split"] != "train":
            continue
        if entry["fold_id"] is None:
            raise CrossValidationError(f"train peptide {row['peptide_id']!r} has no fold_id")
        out.append({**row, "fold_id": entry["fold_id"]})
    return out


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
    graph_encoder_kwargs: dict | None = None,
    peptide_method_kwargs: dict | None = None,
    organism_method_kwargs: dict | None = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    class_balancing_mode: str = "auto",
    class_balancing_fixed_pos_weight: float | None = None,
    device: "torch.device | str | None" = None,
    watch: bool = False,
    on_epoch_end: Callable[[dict], None] | None = None,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """One fold: build_dataset(row_groups=...) -> build_model -> train for
    `epochs` -> evaluate on every eval_groups member. Reproduces
    scripts/EDA/build_dataset_build_model.ipynb's `train_and_evaluate` body
    exactly, generalized to take class-balancing and device as parameters
    (the notebook hardcodes "auto"/None and is CPU-only) -- defaults
    reproduce the notebook's behavior unchanged.

    peptide_method_kwargs/organism_method_kwargs pass straight through to
    build_dataset -- in particular, this is how a caller running multiple
    folds shares one PeptideCLMFeaturizer cache dict across them (each fold
    still constructs a fresh featurizer instance, but the same dict object
    passed via peptide_method_kwargs={"cache": ...} makes embeddings
    computed in an earlier fold get reused rather than recomputed; see
    soamp.features.peptide_featurizers.PeptideCLMFeaturizer's docstring).

    `on_epoch_end`, if given, is called after every epoch with
    {"epoch": 1-based epoch, "fit_loss": mean training loss of that epoch,
    "val_loss": loss on the `val` group, "val_auroc": its AUROC} -- this is
    how callers get loss curves without this module knowing about any
    tracker. Requires a "val" key in `row_groups`; adds one no-grad pass over
    that group per epoch, and nothing at all when left None.

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
        row_groups=row_groups,
        peptide_method=peptide_method,
        peptide_method_kwargs=peptide_method_kwargs,
        organism_method=organism_method,
        organism_method_kwargs=organism_method_kwargs,
    )

    # Re-seed immediately before model construction, not just once at the top
    # of this function: a peptide featurizer's forward pass (e.g.
    # PeptideCLMFeaturizer, uncached) measurably advances torch's global RNG
    # state (verified directly -- a fresh compute leaves torch.rand() at a
    # different point than a cache hit does), which would otherwise make the
    # model's weight initialization depend on how much of build_dataset's
    # internal work happened to consume RNG -- including, concretely,
    # whether a given peptide was a cache hit or miss. Reseeding here
    # decouples model-init/training determinism from that entirely, so
    # `seed` alone controls the model regardless of featurization caching.
    torch.manual_seed(seed)
    model = build_model(
        bundle, architecture=architecture, graph_encoder_kwargs=graph_encoder_kwargs,
        **(architecture_kwargs or {}),
    )

    fit_loader = build_loader(bundle.datasets["fit"], batch_size, shuffle=True)

    fit_labels = [LABEL_TO_INT[row["label"]] for row in row_groups["fit"]]
    pos_weight = resolve_pos_weight(class_balancing_mode, class_balancing_fixed_pos_weight, fit_labels)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=resolved_device) if pos_weight is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer = Trainer(model, optimizer, loss_fn, device=resolved_device)

    if watch:
        # Watches the real model about to be trained -- gradient/parameter
        # histograms and the computation graph populate from its actual
        # forward/backward passes below, not a throwaway preview model.
        watch_model(model)

    val_loader = (
        build_loader(bundle.datasets["val"], batch_size)
        if on_epoch_end is not None else None
    )
    for epoch in range(1, epochs + 1):
        fit_loss = trainer.train_epoch(fit_loader)["loss"]
        if on_epoch_end is not None:
            val_out = trainer.evaluate(val_loader)
            on_epoch_end({
                "epoch": epoch,
                "fit_loss": fit_loss,
                "val_loss": val_out["loss"],
                "val_auroc": compute_binary_metrics(val_out["logits"], val_out["labels"])["auroc"],
            })

    # Each eval_loader has shuffle=False, so its eval_out arrays line up 1:1
    # with row_groups[group] in order -- safe to reconstruct a per-row table.
    metrics_by_group = {}
    group_frames = []
    for group in eval_groups:
        eval_loader = build_loader(bundle.datasets[group], batch_size)
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
