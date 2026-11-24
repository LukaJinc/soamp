"""Laptop, once (and whenever data changes): upload a dataset's inputs to W&B.

Cloud sessions and vast.ai instances have no access to the gitignored data/, so
instances download this artifact instead of being rsynced. Files are stored under
their repo-relative paths (data/dev/...), which `scripts/exp/fetch_data.py`
restores in place.

    python scripts/exp/publish_data.py --dataset dev
Artifact: <entity>/soamp-ops/soamp_<dataset>_data:latest
"""
import argparse
import logging
import os
import sys

import wandb
from dotenv import load_dotenv

from soamp.runs.data_artifact import build_metadata, describe
from soamp.runs.launch_plan import data_files
from soamp.runs.ledger import OPS_PROJECT
from soamp.runs.request import DEFAULT_DATASET
from soamp.utils.config import REPO_ROOT

log = logging.getLogger("publish_data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")

    wanted = data_files(args.dataset)
    present = [f for f in wanted if (REPO_ROOT / f).is_file()]
    for f in wanted:
        log.info("%-60s %s", f, f"{(REPO_ROOT / f).stat().st_size / 1e6:8.2f} MB" if f in present else "(absent: built on the instance)")
    required_missing = [f for f in wanted[:2] if f not in present]  # the required inputs come first
    if required_missing:
        log.error("required input(s) missing locally: %s", required_missing)
        return 2
    # Provenance travels with the artifact: per-file size + sha256, counts, git commit (soamp.runs.data_artifact).
    meta = build_metadata(args.dataset, REPO_ROOT, present, wanted[0])
    log.info("total %.1f MB in %d files | %s", sum(v["bytes"] for v in meta["files"].values()) / 1e6, len(present), describe(meta))
    if meta["git"]["dirty"]:
        log.warning("tracked files have uncommitted changes: the recorded git sha does not fully describe this data")
    if args.dry_run:
        return 0

    run = wandb.init(entity=os.environ.get("WANDB_ENTITY"), project=OPS_PROJECT, job_type="publish_data", name=f"publish-{args.dataset}")
    art = wandb.Artifact(f"soamp_{args.dataset}_data", type="dataset", description=describe(meta), metadata=meta)
    for f in present:
        art.add_file(str(REPO_ROOT / f), name=f)
    run.log_artifact(art, aliases=["latest"]).wait()
    log.info("published %s", art.qualified_name)
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
