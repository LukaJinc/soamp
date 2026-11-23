"""Deterministic analysis of one or more experiments (W&B groups) -> facts.json + tidy metrics.

    python scripts/exp/analyze.py --experiment <group> [--experiment <group2>] [--baseline-run NAME]

Writes reports/experiments/<experiments>/<input_hash>/{facts.json,metrics.csv}. The input hash
is the same one the watcher uses (finished run ids + their cv_results artifact versions), so a
re-run on unchanged results lands in the same directory and the deck step can skip it.
Prints the output directory on the last line. No narrative here: that is the agent's job
(see the analyze-experiment skill), from facts.json only.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import wandb
from dotenv import load_dotenv

from soamp.analysis.facts import analyze
from soamp.analysis.loader import load_group, load_named
from soamp.runs.ledger import Manifest, WandbLedgerBackend
from soamp.runs.status import RunInfo, classify_cell, input_hash
from soamp.runs.watch_backend import WandbStatusBackend
from soamp.utils.config import REPO_ROOT

log = logging.getLogger("analyze")
DEFAULT_BASELINE = "dev_sweep_small_wd1e-3"  # CLAUDE.md's dev reference run


def _failed_cells(experiment: str, finished: set[str], entity: str, now: datetime) -> list[dict]:
    """Cells of the experiment's latest launch that did not finish, with their classified state."""
    rows = [m for m in WandbLedgerBackend(entity).launches() if m.get("experiment") == experiment]
    if not rows:
        return []
    m = Manifest.from_dict(max(rows, key=lambda r: r["created_at"]))
    backend = WandbStatusBackend(entity)
    runs, records = backend.group_runs(experiment), backend.status_records(experiment)
    out = []
    for c in m.cells:
        if c in finished:
            continue
        state = classify_cell(c, runs.get(c), status_records=records, instance_alive=False, past_deadline=m.is_past_deadline(now), now=now)
        if state == "finished":
            continue
        rc = next((rec["cells"][c].get("returncode") for rec in records if c in (rec.get("cells") or {})), None)
        out.append({"cell": c, "state": state, "detail": f"rc={rc}" if rc is not None else ""})
    return out


def _cost(experiments: list[str], entity: str) -> dict | None:
    """Estimated cost (latest launch manifest) and measured training time (instance status records)."""
    launches = WandbLedgerBackend(entity).launches()
    backend = WandbStatusBackend(entity)
    est, secs = 0.0, 0.0
    for e in experiments:
        rows = [m for m in launches if m.get("experiment") == e]
        if not rows:
            continue
        est += float(max(rows, key=lambda r: r["created_at"])["estimate_usd"])
        for rec in backend.status_records(e):
            secs += sum(float(c.get("seconds", 0)) for c in (rec.get("cells") or {}).values())
    return {"estimate_usd": round(est, 2), "train_hours": round(secs / 3600, 3)} if est else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", action="append", required=True)
    ap.add_argument("--baseline-run", default=DEFAULT_BASELINE, help="reference run (by name) to compare against; falls back to the best run if absent")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    now = datetime.now(timezone.utc)
    runs = []
    for exp in args.experiment:
        got = load_group(api, entity, exp)
        log.info("%s: %d finished run(s)", exp, len(got))
        runs += got
    if not runs:
        log.error("no finished runs in %s", args.experiment)
        return 1
    exp_runs = list(runs)  # the experiment's own runs: what the idempotency hash covers (same as the watcher)
    if args.baseline_run and args.baseline_run not in {r.name for r in runs}:
        ref = load_named(api, entity, args.baseline_run)
        if ref is not None:
            log.info("added reference baseline run %s", ref.name)
            runs.append(ref)
    finished = {r.name for r in exp_runs}
    failed = [c for e in args.experiment for c in _failed_cells(e, finished, entity, now)]

    facts, tidy_df = analyze(runs, baseline=args.baseline_run, failed_cells=failed)
    facts["experiments"] = args.experiment
    facts["cost"] = _cost(args.experiment, entity)
    h = input_hash([RunInfo(r.name, "finished", r.run_id, artifact_version=r.artifact_version) for r in exp_runs])
    facts["input_hash"] = h
    out = REPO_ROOT / "reports" / "experiments" / "+".join(args.experiment) / h
    out.mkdir(parents=True, exist_ok=True)
    tidy_df.to_csv(out / "metrics.csv", index=False)
    (out / "facts.json").write_text(json.dumps(facts, indent=2, default=str))
    log.info("ranking: %s", [(r["run"], r["consensus"]) for r in facts["ranking"]])
    log.info("flags: %d run-level, %d global", len(facts["flags"]), len(facts["global_flags"]))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
