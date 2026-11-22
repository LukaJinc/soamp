"""Pull fit-vs-val metrics of every dev-data CV run from wandb for the deck's "Models" chapter.

Per run (wandb groups dev_featurization_cv_v1, dev_featurization_cv_v3, dev_sweep_v1), per fold,
epoch and eval_group (fit / val): MCC, precision, recall, AUROC, accuracy, F1 and loss, then mean/std
over the 5 folds. Read-only on wandb.

Sources, per run:
- runs with a `cv_snapshots` table (the 100-epoch run and the sweep cells): the table itself
  (eval-mode metrics for fit and val every N epochs).
- the 5-epoch run (no snapshots): the final-epoch row-level predictions in its `cv_results` CSV.
  Metrics are recomputed from the logits; the eval-mode loss, which was not stored, is recomputed
  as the trainer computes it: BCE-with-logits with pos_weight = n_neg / n_pos of the fold's fit
  labels, mean of per-batch means (batch size 256, rows in loader order).

Writes reports/dev_experiments_metrics.csv (tidy) and reports/slide_figs/dev_experiments_facts.json.

usage: python scripts/slides/build_dev_experiment_facts.py
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

from soamp.engine.metrics import compute_binary_metrics

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["dev_featurization_cv_v1", "dev_featurization_cv_v3", "dev_sweep_v1", "dev_repr_v1", "dev_class_balance_v1"]
METRICS = ["mcc", "precision", "recall", "auroc", "accuracy", "f1", "loss"]
# Logged only by runs made after the prior-matched metrics were added; used when present.
EXTRA_METRICS = ["f1_matched", "mcc_matched", "pred_active_rate"]
BATCH_SIZE = 256


def weighted_bce_batch_mean(logits: np.ndarray, labels: np.ndarray, pos_weight: float,
                            batch_size: int = BATCH_SIZE) -> float:
    """Mean over batches of the per-batch mean of BCEWithLogitsLoss(pos_weight), as Trainer.evaluate."""
    x = logits.astype(np.float64)
    y = labels.astype(np.float64)
    per_row = pos_weight * y * np.logaddexp(0.0, -x) + (1.0 - y) * np.logaddexp(0.0, x)
    starts = range(0, len(per_row), batch_size)
    return float(np.mean([per_row[i:i + batch_size].mean() for i in starts]))


def metrics_from_rows(rows: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """Per (val_fold_id, eval_group) metrics incl. recomputed loss from row-level predictions.
    `rows` needs: val_fold_id, eval_group, logit, true_label_int (rows in loader order)."""
    out = []
    for fold, fold_rows in rows.groupby("val_fold_id", sort=True):
        fit = fold_rows[fold_rows.eval_group == "fit"]
        pos_weight = float((fit.true_label_int == 0).sum() / (fit.true_label_int == 1).sum())
        for group, g in fold_rows.groupby("eval_group"):
            logits, labels = g["logit"].to_numpy(), g["true_label_int"].to_numpy()
            m = compute_binary_metrics(logits, labels)
            m["loss"] = weighted_bce_batch_mean(logits, labels, pos_weight)
            out.append({"val_fold_id": fold, "epoch": epoch, "eval_group": group, **m})
    return pd.DataFrame(out)


def aggregate(snapshots: pd.DataFrame, run: str, group: str) -> pd.DataFrame:
    """Tidy mean/std over folds: columns run, group, epoch, eval_group, metric, mean, std, n_folds."""
    value_vars = METRICS + [m for m in EXTRA_METRICS if m in snapshots.columns]
    long = snapshots.melt(id_vars=["val_fold_id", "epoch", "eval_group"], value_vars=value_vars,
                          var_name="metric", value_name="value")
    long["value"] = long["value"].astype(float)
    agg = (long.groupby(["epoch", "eval_group", "metric"])["value"]
           .agg(mean="mean", std="std", n_folds="count").reset_index())
    agg.insert(0, "group", group)
    agg.insert(0, "run", run)
    return agg


def load_run_snapshots(run, workdir: str) -> tuple[pd.DataFrame, str]:
    art = next(a for a in run.logged_artifacts() if a.name.startswith("cv_results"))
    path = art.download(root=os.path.join(workdir, run.id))
    if any(f.endswith("cv_snapshots.table.json") for f in os.listdir(path)):
        t = art.get("cv_snapshots")
        snaps = pd.DataFrame(t.data, columns=t.columns)
        snaps["epoch"] = snaps["epoch"].astype(int)
        snaps["val_fold_id"] = snaps["val_fold_id"].astype(int)
        return snaps, "cv_snapshots table"
    csv = next(f for f in os.listdir(path) if f.startswith("cv_results_") and f.endswith(".csv"))
    rows = pd.read_csv(os.path.join(path, csv))
    return metrics_from_rows(rows, epoch=int(run.config.get("loop", {}).get("epochs", 0)) or 5), \
        "row-level predictions (loss recomputed)"


def main() -> None:
    api = wandb.Api()
    workdir = tempfile.mkdtemp()
    frames, runs_meta = [], []
    for group in GROUPS:
        for run in sorted(api.runs("soamp", filters={"group": group}), key=lambda r: r.created_at):
            if run.state != "finished":
                continue
            snaps, source = load_run_snapshots(run, workdir)
            frames.append(aggregate(snaps, run.name, group))
            runs_meta.append({"run": run.name, "group": group, "source": source,
                              "n_parameters": run.summary.get("n_parameters"),
                              "epochs": int(snaps.epoch.max()), "created_at": run.created_at})
    tidy = pd.concat(frames, ignore_index=True)
    out_csv = REPO_ROOT / "reports" / "dev_experiments_metrics.csv"
    tidy.to_csv(out_csv, index=False)
    facts = REPO_ROOT / "reports" / "slide_figs" / "dev_experiments_facts.json"
    facts.write_text(json.dumps({"runs": runs_meta, "metrics": METRICS}, indent=2))
    print(f"wrote {out_csv} ({len(tidy)} rows) and {facts}")
    print(pd.DataFrame(runs_meta).to_string())


if __name__ == "__main__":
    main()
