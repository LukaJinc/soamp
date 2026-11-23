"""Deterministic analysis of CV runs: tidy metrics, ranking, paired comparison, suspicion flags.

Pure functions on DataFrames (no W&B, no plotting), so everything is unit-tested and the
same inputs always give the same ``facts``. Selection metrics follow CLAUDE.md's dev
decision rule: val loss, val MCC and val F1 (AUROC etc. are reported, never selected on).
Epochs are the unit of training length; for tree models one boosting round counts as one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from soamp.utils.config import REPO_ROOT

METRICS = ["mcc", "precision", "recall", "auroc", "accuracy", "f1", "loss"]
EXTRA_METRICS = ["f1_matched", "mcc_matched", "pred_active_rate"]  # only runs made after they were added
SELECTION = (("val_loss", "loss", True), ("val_mcc", "mcc", False), ("val_f1", "f1", False))  # (name, metric, lower_is_better)
FLAGS_PATH = "config/analysis/flags.yaml"
FACTS_VERSION = "1"


class FlagThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fit_val_mcc_gap: float
    val_loss_late_ratio: float
    f1_drift_min_gain: float
    matched_f1_max_gain: float
    fold_noise_mcc_std: float
    param_outlier_ratio: float


def load_thresholds(path: str = FLAGS_PATH) -> FlagThresholds:
    return FlagThresholds(**yaml.safe_load((REPO_ROOT / path).read_text()))


@dataclass
class RunData:
    name: str
    run_id: str
    config: dict  # architecture, peptide_method, organism_method, hyperparameters, seed, class_balancing_mode ...
    n_parameters: float | None
    snapshots: pd.DataFrame  # val_fold_id, epoch, eval_group, <metrics>
    artifact_version: str = ""
    extra: dict = field(default_factory=dict)


def tidy(runs: list[RunData]) -> pd.DataFrame:
    """run, epoch, eval_group, metric, mean, std, n_folds -- mean/std over the CV folds."""
    frames = []
    for r in runs:
        cols = METRICS + [m for m in EXTRA_METRICS if m in r.snapshots.columns]
        cols = [c for c in cols if c in r.snapshots.columns]
        long = r.snapshots.melt(id_vars=["val_fold_id", "epoch", "eval_group"], value_vars=cols, var_name="metric", value_name="value")
        long["value"] = long["value"].astype(float)
        agg = long.groupby(["epoch", "eval_group", "metric"])["value"].agg(mean="mean", std="std", n_folds="count").reset_index()
        agg.insert(0, "run", r.name)
        frames.append(agg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["run", "epoch", "eval_group", "metric", "mean", "std", "n_folds"])


def _at(t: pd.DataFrame, run: str, group: str, metric: str) -> pd.Series:
    s = t[(t.run == run) & (t.eval_group == group) & (t.metric == metric)].set_index("epoch")
    return s["mean"].astype(float).sort_index()


def _std_at(t: pd.DataFrame, run: str, group: str, metric: str, epoch: int) -> float:
    s = t[(t.run == run) & (t.eval_group == group) & (t.metric == metric) & (t.epoch == epoch)]["std"]
    return float(s.iloc[0]) if len(s) and pd.notna(s.iloc[0]) else float("nan")


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


HPARAM_KEYS = ["learning_rate", "weight_decay", "batch_size", "epochs", "projection_dim", "hidden_dims", "organism_embed_dim",
               "num_attention_heads", "num_attention_layers", "max_iter", "max_leaf_nodes", "min_samples_leaf",
               "l2_regularization", "max_features"]


def run_facts(r: RunData, t: pd.DataFrame, th: FlagThresholds) -> tuple[dict, list[dict]]:
    """Per-run summary + the flags this run triggers."""
    flags: list[dict] = []
    last = int(t[t.run == r.name].epoch.max())
    first = int(t[t.run == r.name].epoch.min())

    def val(group: str, metric: str, epoch: int = last):
        s = _at(t, r.name, group, metric)
        return float(s.loc[epoch]) if epoch in s.index else None

    stats = {}
    for group in ("fit", "val"):
        for m in METRICS + EXTRA_METRICS:
            v = val(group, m)
            if v is not None:
                stats[f"{group}_{m}"] = {"mean": _r(v), "std": _r(_std_at(t, r.name, group, m, last))}
    best = {}
    for name, metric, lower in SELECTION:
        s = _at(t, r.name, "val", metric)
        if len(s):
            ep = int(s.idxmin() if lower else s.idxmax())
            best[name] = {"epoch": ep, "value": _r(s.loc[ep])}
    gap = None
    if "fit_mcc" in stats and "val_mcc" in stats:
        gap = _r(stats["fit_mcc"]["mean"] - stats["val_mcc"]["mean"])
    facts = {
        "name": r.name, "run_id": r.run_id, "architecture": r.config.get("architecture"),
        "peptide": r.config.get("peptide_method"), "organism": r.config.get("organism_method"),
        "seed": r.config.get("seed"), "class_balancing": r.config.get("class_balancing_mode"),
        "n_parameters": _r(r.n_parameters, 0), "last_epoch": last,
        "hparams": {k: r.config[k] for k in HPARAM_KEYS if k in r.config and r.config[k] is not None},
        "last": stats, "best": best, "mcc_gap_fit_minus_val": gap,
    }

    def flag(kind: str, severity: str, message: str) -> None:
        flags.append({"id": f"{r.name}:{kind}", "run": r.name, "kind": kind, "severity": severity, "message": message})

    if gap is not None and gap > th.fit_val_mcc_gap:
        flag("overfit_gap", "warn", f"fit MCC exceeds val MCC by {gap:.3f} at epoch {last} (threshold {th.fit_val_mcc_gap})")
    vl = _at(t, r.name, "val", "loss")
    if len(vl) > 2:
        ep_min = int(vl.idxmin())
        if ep_min < 0.5 * last and vl.iloc[-1] > th.val_loss_late_ratio * vl.min():
            flag("val_loss_rises", "warn", f"val loss is lowest at epoch {ep_min} ({vl.min():.3f}) and rises to {vl.iloc[-1]:.3f} by epoch {last}")
    f1, f1m = _at(t, r.name, "val", "f1"), _at(t, r.name, "val", "f1_matched")
    if len(f1) > 1 and len(f1m) > 1 and first != last:
        d, dm = float(f1.iloc[-1] - f1.iloc[0]), float(f1m.iloc[-1] - f1m.iloc[0])
        if d > th.f1_drift_min_gain and dm < th.matched_f1_max_gain:
            flag("calibration_drift", "warn", f"val F1@0.5 rose {d:+.3f} but prior-matched F1 only {dm:+.3f}: calibration drift, not learning")
    sd = stats.get("val_mcc", {}).get("std")
    if sd is not None and not np.isnan(sd) and sd > th.fold_noise_mcc_std:
        flag("fold_noise", "info", f"val MCC std over folds is {sd:.3f}; small differences are not reliable")
    return facts, flags


def _rank(values: dict[str, float | None], lower: bool) -> dict[str, int | None]:
    """Competition ranking (1, 1, 3): equal values share a rank, so input order never decides."""
    ok = {k: round(v, 6) for k, v in values.items() if v is not None}
    out: dict[str, int | None] = {k: None for k in values}
    for k, v in ok.items():
        out[k] = 1 + sum((o < v) if lower else (o > v) for o in ok.values())
    return out


def fold_values(r: RunData, group: str, metric: str, epoch: int) -> pd.Series:
    s = r.snapshots[(r.snapshots.eval_group == group) & (r.snapshots.epoch == epoch)]
    return s.set_index("val_fold_id")[metric].astype(float).sort_index()


def analyze(runs: list[RunData], *, baseline: str | None = None, thresholds: FlagThresholds | None = None,
            failed_cells: list[dict] | None = None) -> tuple[dict, pd.DataFrame]:
    """Facts dict + the tidy metrics table for these runs (the first run listed as `baseline`
    is excluded from nothing: it is ranked with the rest)."""
    th = thresholds or load_thresholds()
    t = tidy(runs)
    per_run, flags = {}, []
    for r in runs:
        f, fl = run_facts(r, t, th)
        per_run[r.name] = f
        flags += fl

    # ranking on the three selection metrics at each run's final epoch; val loss only comparable
    # among runs trained with the same class weighting (weighted vs plain BCE differ by construction)
    names = [r.name for r in runs]
    cb = {r.name: r.config.get("class_balancing_mode") for r in runs}
    mixed_cb = len(set(cb.values())) > 1
    ranks = {}
    for sel, metric, lower in SELECTION:
        vals = {n: per_run[n]["last"].get(f"val_{metric}", {}).get("mean") for n in names}
        if sel == "val_loss" and mixed_cb:
            vals = {n: None for n in names}
        ranks[sel] = _rank(vals, lower)
    ranking = []
    for n in names:
        rs = [ranks[s][n] for s, _, _ in SELECTION if ranks[s][n] is not None]
        ranking.append({"run": n, **{f"rank_{s}": ranks[s][n] for s, _, _ in SELECTION}, "consensus": _r(np.mean(rs), 2) if rs else None})
    ranking.sort(key=lambda x: (x["consensus"] is None, x["consensus"] if x["consensus"] is not None else 0, x["run"]))

    by_name = {r.name: r for r in runs}
    base = baseline if baseline in by_name else (ranking[0]["run"] if ranking else None)
    vs = []
    if base is not None:
        b = by_name[base]
        for r in runs:
            if r.name == base:
                continue
            row = {"run": r.name, "baseline": base}
            for sel, metric, lower in SELECTION:
                if sel == "val_loss":
                    continue
                d = (fold_values(r, "val", metric, per_run[r.name]["last_epoch"])
                     - fold_values(b, "val", metric, per_run[base]["last_epoch"])).dropna()
                if d.empty:
                    continue
                noise = max(per_run[r.name]["last"].get(f"val_{metric}", {}).get("std") or 0.0,
                            per_run[base]["last"].get(f"val_{metric}", {}).get("std") or 0.0)
                verdict = "tie" if abs(d.mean()) < noise else ("better" if d.mean() > 0 else "worse")
                row[f"delta_{metric}"] = {"mean": _r(d.mean()), "folds_better": int((d > 0).sum()), "folds_worse": int((d < 0).sum()),
                                          "n_folds": int(len(d)), "noise": _r(noise), "verdict": verdict}
            vs.append(row)

    global_flags = []
    seeds = {r.config.get("seed") for r in runs}
    if len(seeds) <= 1:
        global_flags.append({"id": "single_seed", "severity": "info", "message": "every run uses one seed: differences below the fold std are not evidence"})
    if mixed_cb:
        global_flags.append({"id": "mixed_class_weighting", "severity": "warn", "message": "runs differ in class weighting: val loss is not comparable across them and is excluded from the ranking"})
    if len({per_run[n]["last_epoch"] for n in names}) > 1:
        global_flags.append({"id": "epoch_budgets_differ", "severity": "info", "message": "runs were trained for different numbers of epochs/rounds; each is compared at its own final epoch"})
    pars = {n: per_run[n]["n_parameters"] for n in names if per_run[n]["n_parameters"]}
    for n, p in pars.items():
        others = [v for k, v in pars.items() if k != n]
        if others and p > th.param_outlier_ratio * float(np.median(others)):
            flags.append({"id": f"{n}:param_outlier", "run": n, "kind": "param_outlier", "severity": "info",
                          "message": f"{int(p)} parameters is >{th.param_outlier_ratio:g}x the median of the other runs"})
    for c in failed_cells or []:
        global_flags.append({"id": f"failed:{c['cell']}", "severity": "warn", "message": f"cell {c['cell']} {c['state']}" + (f" ({c['detail']})" if c.get("detail") else "")})

    facts = {
        "version": FACTS_VERSION, "n_runs": len(runs), "baseline": base, "runs": [per_run[n] for n in names],
        "ranking": ranking, "vs_baseline": vs, "flags": flags, "global_flags": global_flags,
        "failed_cells": failed_cells or [],
    }
    return facts, t


def facts_hash(facts: dict) -> str:
    return hashlib.sha1(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()[:8]
