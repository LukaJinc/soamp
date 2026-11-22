"""Production vs dev: val metrics of the two 100-epoch PeptideCLM x vocab CV runs at chosen epochs.

Row-level val predictions (`cv_val_predictions.csv.gz` in each run's `cv_results` artifact) are
scored per fold with soamp.engine.metrics.compute_binary_metrics (0.5 threshold AND the prior-matched
threshold), then averaged over folds. Val loss comes from wandb's per-epoch `fold/N/val_loss` history.
Read-only on wandb. Writes reports/prod_vs_dev_metrics.csv.

usage: python scripts/slides/build_prod_vs_dev_facts.py
"""
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
# dataset label -> (wandb group, run name, classification csv, epochs, class weight override)
RUNS = {"prod": ("featurization_grid_cv_v3", "peptideclm_vocab_cv_long", "data/mic_classification_dataset.csv", (5, 100), None),
        "dev": ("dev_featurization_cv_v3", "dev_peptideclm_vocab_cv_long", "data/dev/mic_classification_dataset.csv", (5, 100), None),
        # class-balancing check: same small model + weight decay 1e-3, auto class weight vs none
        "dev_small_auto": ("dev_sweep_v1", "dev_sweep_small_wd1e-3", "data/dev/mic_classification_dataset.csv", (4, 40), None),
        "dev_small_none": ("dev_class_balance_v1", "dev_cb_none_small_wd1e-3", "data/dev/mic_classification_dataset.csv", (4, 40), 1.0)}
EPOCHS = (5, 100)
KEYS = ["precision", "recall", "f1", "f1_matched", "mcc", "mcc_matched", "auroc", "pred_active_rate"]


def val_metrics_at_epochs(preds: pd.DataFrame, epochs=EPOCHS) -> pd.DataFrame:
    """preds: columns val_fold_id, epoch, true_label_int, logit. Mean over folds of each metric,
    plus the true active rate. One row per epoch."""
    out = []
    for epoch in epochs:
        per_fold = []
        for _, g in preds[preds.epoch == epoch].groupby("val_fold_id"):
            m = compute_binary_metrics(g["logit"].to_numpy(), g["true_label_int"].to_numpy())
            m["true_active_rate"] = float(g["true_label_int"].mean())
            per_fold.append(m)
        mean = pd.DataFrame(per_fold).mean()
        out.append({"epoch": epoch, **{k: float(mean[k]) for k in [*KEYS, "true_active_rate"]}})
    return pd.DataFrame(out)


def class_weight(csv_path: Path) -> dict:
    d = pd.read_csv(csv_path)
    t = d[d.split == "train"]
    a = float((t.label == "active").mean())
    return {"train_rows": len(t), "active_share": a, "pos_weight": (1 - a) / a}


def main() -> None:
    api = wandb.Api()
    rows = []
    for name, (group, run_name, csv_rel, epochs, pw_override) in RUNS.items():
        run = [r for r in api.runs("soamp", filters={"group": group}) if r.name == run_name and r.state == "finished"][-1]
        art = next(a for a in run.logged_artifacts() if a.name.startswith("cv_results"))
        path = art.download(root=tempfile.mkdtemp())
        preds = pd.read_csv(os.path.join(path, "cv_val_predictions.csv.gz"))
        res = val_metrics_at_epochs(preds, epochs)
        hist = run.history(samples=5000, pandas=True).dropna(subset=["epoch"]).groupby("epoch").mean(numeric_only=True)
        vl = hist[[c for c in hist.columns if c.startswith("fold/") and c.endswith("/val_loss")]].mean(axis=1)
        res["val_loss"] = [float(vl.loc[e]) for e in res.epoch]
        res["n_fit_per_fold"] = run.summary.get("n_fit")
        for k, v in class_weight(REPO_ROOT / csv_rel).items():
            res[k] = v
        if pw_override is not None:
            res["pos_weight"] = pw_override
        res.insert(0, "dataset", name)
        rows.append(res)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(REPO_ROOT / "reports" / "prod_vs_dev_metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
