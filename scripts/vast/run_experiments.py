"""On-instance runner: execute every pending CV experiment in a run spec.

Thin orchestration only -- `pipeline/train_cv.py` stays the single training
entrypoint; this loops over it. One failed cell does not stop the others.
Exit code is non-zero if any cell failed, so remote_main.sh can pick the
destroy path. `--dry-run` resolves and prints the plan without training.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time

from soamp.runs.spec import (
    RunSpec,
    describe_experiment,
    pending_experiments,
    pending_prepare_steps,
    required_artifacts_missing,
    resolve_config_paths,
)
from soamp.utils.config import REPO_ROOT, load_config

log = logging.getLogger("run_experiments")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spec = load_config(args.spec, RunSpec)
    experiments = [describe_experiment(p) for p in resolve_config_paths(spec)]
    finished: set[str] = set()
    if os.environ.get("WANDB_API_KEY"):
        from soamp.runs.wandb_state import finished_run_names

        for group in sorted({e.wandb_group for e in experiments if e.wandb_group}):
            try:
                finished |= finished_run_names(group)
            except Exception as e:  # never block training on a status lookup
                log.warning("could not read finished runs of group %s from wandb: %s", group, e)
    pending = pending_experiments(experiments, finished)
    log.info("%d experiments, %d pending", len(experiments), len(pending))
    for e in experiments:
        log.info("  %s %s", "TODO" if e in pending else "done", e.exp_id)

    missing = required_artifacts_missing(spec)
    if missing:
        log.error("required artifacts missing: %s", [str(p) for p in missing])
        return 2
    steps = pending_prepare_steps(spec)
    for st in pending_prepare_steps(spec):
        log.info("prepare step pending: %s", " ".join(st.cmd))
    if args.dry_run:
        return 0

    for st in steps:
        rc = subprocess.run(st.cmd, cwd=REPO_ROOT).returncode
        if rc != 0 or not all((REPO_ROOT / p).is_file() for p in st.produces):
            log.error("prepare step failed (rc=%d): %s", rc, " ".join(st.cmd))
            return 3

    status: dict[str, dict] = {}
    out = REPO_ROOT / "reports" / "vast_run_status.json"
    out.parent.mkdir(exist_ok=True)
    for e in pending:
        start = time.time()
        # TODO(skeleton): per-cell upload of reports/*<exp_id>* to wandb so
        # results survive instance destruction (train_cv.py already uploads cv_results).
        rc = subprocess.run(
            [sys.executable, "pipeline/train_cv.py", "--config", str(e.config_path)],
            cwd=REPO_ROOT,
        ).returncode
        status[e.exp_id] = {"returncode": rc, "seconds": round(time.time() - start)}
        log.info("%s finished rc=%d", e.exp_id, rc)
        # written after every cell, so a timeout/kill still leaves the completed cells on record
        out.write_text(json.dumps(status, indent=2))
    return 1 if any(s["returncode"] for s in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
