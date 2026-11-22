"""Compare the CV runs of one wandb group from their saved `cv_snapshots` tables (read-only).

Decisions are made on **val loss, val MCC and val F1** (AUROC is logged and shown, but is not a
selection metric). Per run, from the mean over folds at each snapshot epoch: the best epoch and
value of each decision metric, the last epoch's values, and the fit-minus-val gaps of the three.

NOTE: a best epoch is picked on the same held-out folds it is reported on, so the "best" columns
are optimistic. Compare cells on them, but prefer the last-epoch columns and confirm the chosen
setting on the untouched test split.

usage: python scripts/vast/summarize_sweep.py --group dev_repr_v1 [--group dev_sweep_v1 ...] [--project soamp]
"""
import argparse
import tempfile

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

DECISION = {"loss": "min", "mcc": "max", "f1": "max"}


def summarize_snapshots(snaps: pd.DataFrame) -> dict:
    """snaps: cv_snapshots table (val_fold_id, epoch, eval_group, loss, mcc, f1, auroc, ...)."""
    snaps = snaps.copy()
    snaps["epoch"] = snaps["epoch"].astype(int)
    metrics = [*DECISION, "auroc", *[m for m in ("f1_matched", "pred_active_rate") if m in snaps.columns]]
    for m in metrics:
        snaps[m] = snaps[m].astype(float)
    mean = snaps.groupby(["eval_group", "epoch"])[metrics].mean()
    val, fit = mean.loc["val"], mean.loc["fit"]
    last = val.index.max()
    out = {}
    for m, how in DECISION.items():
        epoch = int(val[m].idxmin() if how == "min" else val[m].idxmax())
        out[f"best_{m}"] = round(float(val.loc[epoch, m]), 4)
        out[f"@ep_{m}"] = epoch
    out["last_ep"] = int(last)
    for m in DECISION:
        out[f"last_{m}"] = round(float(val.loc[last, m]), 4)
    for m in DECISION:
        out[f"gap_{m}"] = round(float(val.loc[last, m] - fit.loc[last, m]), 4)  # val minus fit
    out["last_auroc"] = round(float(val.loc[last, "auroc"]), 4)
    # present only for runs logged with the prior-matched metrics: F1 at a threshold that predicts the
    # true active rate (calibration-free) and the share predicted active at 0.5
    for m in ("f1_matched", "pred_active_rate"):
        if m in val.columns:
            out[f"last_{m}"] = round(float(val.loc[last, m]), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--project", default="soamp")
    args = parser.parse_args()

    api = wandb.Api()
    rows = []
    for group in args.group:
        for run in sorted(api.runs(args.project, filters={"group": group}), key=lambda r: r.created_at):
            if run.state != "finished":
                rows.append({"run": run.name, "state": run.state})
                continue
            art = next((a for a in run.logged_artifacts() if a.name.startswith("cv_results")), None)
            if art is None:
                rows.append({"run": run.name, "state": "no cv_results artifact"})
                continue
            art.download(root=tempfile.mkdtemp())
            table = art.get("cv_snapshots")
            if table is None:
                rows.append({"run": run.name, "state": "no cv_snapshots table"})
                continue
            rows.append({"run": run.name, "state": "finished",
                         **summarize_snapshots(pd.DataFrame(table.data, columns=table.columns))})
    df = pd.DataFrame(rows).set_index("run")
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(df.to_string())


if __name__ == "__main__":
    main()
