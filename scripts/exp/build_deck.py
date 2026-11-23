"""Validate the agent's narrative, render figures + deck, publish the report (idempotent).

    python scripts/exp/build_deck.py --dir reports/experiments/<exp>/<hash> [--narrative narrative.json] [--no-publish]
    python scripts/exp/build_deck.py --failure <experiment>      # nothing finished: short failure deck

Writes into --dir: figures/*.png, deck_spec.json (what the Slides artifact is made from) and
deck.pdf, then logs them as W&B artifact ``exp_report-<experiment>`` (project soamp-ops) with
metadata input_hash. If that hash is already published the step prints UNCHANGED and does
nothing, so on-demand and scheduled triggers can both run it. Exit 2 = narrative rejected
(problems printed, fix narrative.json and re-run).
"""
import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from soamp.analysis.deck import build_spec, write_pdf, write_spec
from soamp.analysis.figures import FIGURES, render_all
from soamp.analysis.narrative import validate_narrative
from soamp.runs.ledger import OPS_PROJECT
from soamp.utils.config import REPO_ROOT

log = logging.getLogger("build_deck")


def _artifact_name(experiments: list[str]) -> str:
    return "exp_report-" + re.sub(r"[^A-Za-z0-9_.-]", "-", "__".join(experiments))


def _published(entity: str, name: str, input_hash: str) -> bool:
    import wandb

    try:
        return any(a.metadata.get("input_hash") == input_hash for a in wandb.Api().artifacts("exp_report", f"{entity}/{OPS_PROJECT}/{name}"))
    except Exception:
        return False


def _publish(entity: str, name: str, input_hash: str, experiments: list[str], d: Path) -> str:
    import wandb

    run = wandb.init(entity=entity, project=OPS_PROJECT, job_type="exp_report", name=f"report-{name[11:]}-{input_hash}", group=experiments[0])
    art = wandb.Artifact(name, type="exp_report", metadata={"input_hash": input_hash, "experiments": experiments})
    for f in ("facts.json", "narrative.json", "deck_spec.json", "deck.pdf"):
        if (d / f).is_file():
            art.add_file(str(d / f), name=f)
    if (d / "figures").is_dir():
        art.add_dir(str(d / "figures"), name="figures")
    run.log_artifact(art).wait()
    ref = art.qualified_name
    run.finish()
    return ref


def _failure_facts(experiment: str, entity: str) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "exp"))
    from analyze import _failed_cells  # noqa: E402  (script module, not a package)

    from soamp.runs.ledger import Manifest, WandbLedgerBackend

    rows = [m for m in WandbLedgerBackend(entity).launches() if m.get("experiment") == experiment]
    m = Manifest.from_dict(max(rows, key=lambda r: r["created_at"]))
    failed = _failed_cells(experiment, set(), entity, datetime.now(timezone.utc))
    return {"experiments": [experiment], "n_runs": 0, "runs": [], "ranking": [], "flags": [], "global_flags": [],
            "failed_cells": failed, "input_hash": f"failed-{m.created_at[:19]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--narrative", type=Path, help="default: <dir>/narrative.json (optional)")
    ap.add_argument("--failure", metavar="EXPERIMENT")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")
    import wandb

    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity

    if args.failure:
        facts = _failure_facts(args.failure, entity)
        d = REPO_ROOT / "reports" / "experiments" / args.failure / facts["input_hash"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "facts.json").write_text(json.dumps(facts, indent=2))
        tidy = pd.DataFrame()
    else:
        if args.dir is None:
            ap.error("--dir (from analyze.py) or --failure is required")
        d = args.dir
        facts = json.loads((d / "facts.json").read_text())
        tidy = pd.read_csv(d / "metrics.csv")
    experiments = facts["experiments"]
    name, h = _artifact_name(experiments), facts["input_hash"]

    if not args.no_publish and _published(entity, name, h):
        print(f"UNCHANGED: {name} input {h} already published")
        return 0

    narrative = None
    npath = args.narrative or (d / "narrative.json")
    if npath.is_file():
        raw = json.loads(npath.read_text())
        narrative, problems = validate_narrative(raw, facts, FIGURES)
        if problems:
            print("NARRATIVE REJECTED:\n  - " + "\n  - ".join(problems))
            return 2
        if npath != d / "narrative.json":
            (d / "narrative.json").write_text(npath.read_text())
    else:
        log.warning("no narrative.json: building the facts-only deck")

    if facts["runs"]:
        render_all(facts, tidy, d / "figures")
    slides = build_spec(facts, narrative)
    write_spec(slides, d / "deck_spec.json")
    write_pdf(slides, d / "figures", d / "deck.pdf")
    log.info("deck: %d slides -> %s", len(slides), d / "deck.pdf")
    if args.no_publish:
        print(f"BUILT {d}")
        return 0
    print(f"PUBLISHED {_publish(entity, name, h, experiments, d)} dir={d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
