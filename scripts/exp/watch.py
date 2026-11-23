"""One watcher tick: refresh every recent experiment, run the janitor, print actions.

Run by the hourly scheduled cloud task (and usable by hand). Deterministic: it decides,
the agent only acts on the lines it prints.

    ANALYZE <experiment>        all cells terminal, new results -> follow the analyze-experiment skill
    RETRY <experiment>          cells lost to infrastructure -> python scripts/exp/launch.py --retry <experiment> --go
    REPORT_FAILURE <experiment> nothing finished -> deliver a failure report (see the skill)
    JANITOR destroyed <id> ...  an overdue/orphaned soamp-* instance was destroyed
    STATUS ...                  informational, never needs action

Needs WANDB_API_KEY, WANDB_ENTITY (optional locally), VAST_API_KEY (cloud).
    python scripts/exp/watch.py [--dry-run] [--max-age-days 7]
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from soamp.runs.ledger import Ledger, Manifest, WandbLedgerBackend, _utc
from soamp.runs.status import decide, janitor_targets
from soamp.runs.vast import VastClient, load_blocked_host_ids
from soamp.runs.watch_backend import WandbStatusBackend
from soamp.utils.config import REPO_ROOT

log = logging.getLogger("watch")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report janitor targets without destroying")
    ap.add_argument("--max-age-days", type=float, default=7.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(REPO_ROOT / ".env")

    now = datetime.now(timezone.utc)
    backend = WandbStatusBackend()
    launches = WandbLedgerBackend(backend.entity).launches()
    manifests = [Manifest.from_dict(d) for d in launches]
    client = VastClient(blocked_host_ids=load_blocked_host_ids())
    instances = client.list_instances()

    latest: dict[str, Manifest] = {}
    for m in sorted(manifests, key=lambda m: m.created_at):
        latest[m.experiment] = m  # the newest launch of an experiment decides its state
    recent = [m for m in latest.values() if now - _utc(m.created_at) < timedelta(days=args.max_age_days)]

    for m in recent:
        d = decide(
            m, runs=backend.group_runs(m.experiment), status_records=backend.status_records(m.experiment),
            instances=instances, launches=launches, reports_done=backend.reports_done(m.experiment), now=now,
        )
        counts = {s: sum(v == s for v in d.cells.values()) for s in sorted(set(d.cells.values()))}
        print(f"STATUS {m.experiment}: {d.state} {counts} -> {d.action}{(' (' + d.detail + ')') if d.detail else ''}", flush=True)
        if d.action != "NONE":
            print(f"{d.action} {m.experiment}", flush=True)

    for iid, reason in janitor_targets(instances, manifests, now, time.time()):
        if args.dry_run:
            print(f"JANITOR would destroy {iid}: {reason}", flush=True)
        else:
            client.destroy_instance(iid)
            print(f"JANITOR destroyed {iid}: {reason}", flush=True)
    if not recent and not instances:
        print("STATUS nothing open", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
