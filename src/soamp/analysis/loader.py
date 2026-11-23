"""Load CV runs from W&B into ``RunData`` (thin IO around soamp.analysis.facts).

Uses each run's *own* ``cv_results`` artifact (matched by name prefix on the run's
logged artifacts): every run logs an artifact with that same name, so ``cv_results:latest``
is project-wide and would silently return whichever run logged last.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from soamp.analysis.facts import RunData
from soamp.engine.metrics import compute_binary_metrics

PROJECT = "soamp"


def snapshots_from_rows(rows: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """Fallback for runs without a cv_snapshots table: final-epoch metrics recomputed from the
    row-level logits (no loss -- it was not stored). Needs val_fold_id, eval_group, logit, true_label_int."""
    out = []
    for (fold, group), g in rows.groupby(["val_fold_id", "eval_group"], sort=True):
        out.append({"val_fold_id": int(fold), "epoch": epoch, "eval_group": group,
                    **compute_binary_metrics(g["logit"].to_numpy(), g["true_label_int"].to_numpy())})
    return pd.DataFrame(out)


def load_run(run, workdir: str | None = None) -> RunData:
    workdir = workdir or tempfile.mkdtemp()
    art = next(a for a in run.logged_artifacts() if a.name.startswith("cv_results"))
    path = art.download(root=os.path.join(workdir, run.id))
    if any(f.endswith("cv_snapshots.table.json") for f in os.listdir(path)):
        t = art.get("cv_snapshots")
        snaps = pd.DataFrame(t.data, columns=t.columns)
        snaps["epoch"] = snaps["epoch"].astype(int)
        snaps["val_fold_id"] = snaps["val_fold_id"].astype(int)
    else:
        csv = next(f for f in os.listdir(path) if f.startswith("cv_results_") and f.endswith(".csv"))
        epochs = int(run.config.get("epochs") or run.config.get("max_iter") or 0) or 1
        snaps = snapshots_from_rows(pd.read_csv(os.path.join(path, csv)), epoch=epochs)
    return RunData(
        name=run.name, run_id=run.id, config=dict(run.config), n_parameters=run.summary.get("n_parameters"),
        snapshots=snaps, artifact_version=art.name,
    )


def load_group(api, entity: str, group: str, workdir: str | None = None) -> list[RunData]:
    runs = sorted((r for r in api.runs(f"{entity}/{PROJECT}", filters={"group": group}) if r.state == "finished"), key=lambda r: r.created_at)
    return [load_run(r, workdir) for r in runs]


def load_named(api, entity: str, name: str, workdir: str | None = None) -> RunData | None:
    """A finished run by name from anywhere in the project (e.g. a reference baseline)."""
    hits = sorted((r for r in api.runs(f"{entity}/{PROJECT}", filters={"display_name": name}) if r.state == "finished"), key=lambda r: r.created_at)
    return load_run(hits[-1], workdir) if hits else None
