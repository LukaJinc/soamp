"""Instance-side: durable completion record, uploaded before the instance is destroyed.

A destroyed vast.ai instance leaves nothing behind, so remote_main.sh's EXIT trap runs
this to log {rc, per-cell status, log tail} to W&B (project soamp-ops). Imports only
stdlib + wandb so it also works when the venv setup itself failed (rc=10).

usage: python scripts/exp/upload_status.py <rc>
env:   WANDB_API_KEY, WANDB_ENTITY, SOAMP_EXPERIMENT, SOAMP_SHARD, CONTAINER_ID
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_TAIL_LINES = 100


def build_record(rc: int, root: Path = ROOT) -> dict:
    status_file = root / "reports" / "vast_run_status.json"
    log_file = root / "reports" / "vast_remote.log"
    cells = json.loads(status_file.read_text()) if status_file.is_file() else {}
    tail = log_file.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:] if log_file.is_file() else []
    return {
        "experiment": os.environ.get("SOAMP_EXPERIMENT", "unknown"),
        "shard": os.environ.get("SOAMP_SHARD", "0"),
        "instance_id": os.environ.get("CONTAINER_ID", ""),
        "rc": rc,
        "cells": cells,  # {exp_id: {returncode, seconds}}; empty if the runner never started
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "log_tail": tail,
    }


def main() -> int:
    import wandb

    rc = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    rec = build_record(rc)
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"), project="soamp-ops", job_type="instance_status",
        name=f"status-{rec['experiment']}-{rec['shard']}", group=rec["experiment"],
        config={k: v for k, v in rec.items() if k != "log_tail"},
    )
    art = wandb.Artifact(f"run_status-{rec['experiment']}-{rec['shard']}", type="run_status")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "status.json"
        p.write_text(json.dumps(rec, indent=2))
        art.add_file(str(p))
        run.log_artifact(art).wait()
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
