"""Completion detection, retry decisions and the instance janitor -- pure logic.

Cloud sessions are ephemeral, so "is experiment X done?" is answered from durable
facts only: the W&B runs of the experiment's group, the status record each instance
uploads before it self-destructs (project soamp-ops), the launch manifest, and
`vastai show instances`. `scripts/exp/watch.py` gathers those facts through thin
backends and calls ``decide``; everything here is deterministic and unit-tested.

Cell states:   finished | running | crashed | stale | missing | lost
  finished  W&B run finished                      (terminal)
  crashed   W&B run crashed/failed/killed, or the instance reported a non-zero rc for it
  stale     W&B run still "running" but no heartbeat for `stale_after`, or past the deadline
  missing   no W&B run yet and still inside the deadline: booting / queued  (NOT terminal)
  lost      no W&B run and the deadline passed, or the instance already ended without it
  running   W&B run running with a fresh heartbeat (NOT terminal)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from soamp.runs.ledger import Manifest, _utc
from soamp.runs.vast import VastInstance

TERMINAL = frozenset({"finished", "crashed", "lost", "stale"})
STALE_AFTER = timedelta(minutes=20)
MAX_ATTEMPTS = 2  # one automatic retry, infra failures only
INFRA_RCS = frozenset({10, 11, 12})  # setup failed / GPU unusable / data fetch failed (remote_main.sh)
ORPHAN_GRACE_SECONDS = 15 * 60  # an instance may exist briefly before its manifest is recorded


@dataclass(frozen=True)
class RunInfo:
    name: str
    state: str  # W&B run state
    run_id: str = ""
    heartbeat_at: datetime | None = None
    artifact_version: str = ""  # version of the run's cv_results artifact, for the input hash


def classify_cell(
    exp_id: str, run: RunInfo | None, *, status_records: list[dict], instance_alive: bool,
    past_deadline: bool, now: datetime, stale_after: timedelta = STALE_AFTER,
) -> str:
    if run is not None:
        if run.state == "finished":
            return "finished"
        if run.state in ("crashed", "failed", "killed"):
            return "crashed"
        if past_deadline or (run.heartbeat_at is not None and now - run.heartbeat_at > stale_after):
            return "stale"
        return "running"
    reported = {k: v for rec in status_records for k, v in (rec.get("cells") or {}).items()}
    if exp_id in reported and reported[exp_id].get("returncode", 0) != 0:
        return "crashed"  # the cell process failed before it ever logged a run
    if status_records and not instance_alive:
        return "lost"  # instance ended (and reported) without running this cell
    return "lost" if past_deadline else "missing"


def experiment_state(states: dict[str, str]) -> Literal["running", "complete", "partial", "failed"]:
    if any(s not in TERMINAL for s in states.values()):
        return "running"
    done = sum(s == "finished" for s in states.values())
    return "complete" if done == len(states) else ("partial" if done else "failed")


def input_hash(runs: list[RunInfo], code_version: str = "1") -> str:
    """Identity of an analysis input: which runs, at which artifact versions."""
    payload = sorted((r.run_id or r.name, r.artifact_version) for r in runs if r.state == "finished")
    return hashlib.sha1(json.dumps([payload, code_version]).encode()).hexdigest()[:8]


def failure_key(manifest: Manifest) -> str:
    """Marker stored (as an exp_report's input_hash) once a failure report was delivered."""
    return f"failed-{manifest.created_at[:19]}"


def attempts(launches: list[dict], experiment: str) -> int:
    return sum(1 for m in launches if m.get("experiment") == experiment)


@dataclass
class Decision:
    experiment: str
    state: str
    cells: dict[str, str]
    action: Literal["NONE", "ANALYZE", "RETRY", "REPORT_FAILURE"] = "NONE"
    detail: str = ""
    retry_cells: list[str] = field(default_factory=list)


def _is_infra_failure(exp_id: str, state: str, status_records: list[dict]) -> bool:
    if state not in ("lost", "missing"):
        return False  # a crashed/finished cell is a training outcome, never retried
    if not status_records:
        return True  # the instance vanished without a trace
    return any(rec.get("rc") in INFRA_RCS for rec in status_records) or all(
        exp_id not in (rec.get("cells") or {}) for rec in status_records
    )


def decide(
    manifest: Manifest, *, runs: dict[str, RunInfo], status_records: list[dict], instances: list[VastInstance],
    launches: list[dict], reports_done: set[str], now: datetime,
) -> Decision:
    past_deadline = manifest.is_past_deadline(now)
    alive = any(i.id in manifest.instance_ids for i in instances)
    states = {
        c: classify_cell(c, runs.get(c), status_records=status_records, instance_alive=alive, past_deadline=past_deadline, now=now)
        for c in manifest.cells
    }
    state = experiment_state(states)
    d = Decision(manifest.experiment, state, states)
    if state == "running":
        return d
    retryable = [c for c, s in states.items() if _is_infra_failure(c, s, status_records)]
    if retryable and attempts(launches, manifest.experiment) < MAX_ATTEMPTS and manifest.request:
        d.action, d.retry_cells = "RETRY", retryable
        d.detail = f"{len(retryable)} cell(s) lost to infrastructure; attempt {attempts(launches, manifest.experiment) + 1}/{MAX_ATTEMPTS}"
        return d
    finished = [runs[c] for c, s in states.items() if s == "finished"]
    if not finished:
        key = failure_key(manifest)
        if key in reports_done:
            d.detail = "failure already reported"
            return d
        d.action, d.detail = "REPORT_FAILURE", f"no cell finished; mark reported with input_hash={key}"
        return d
    h = input_hash(finished)
    if h in reports_done:
        d.detail = f"already analyzed (input {h})"
        return d
    d.action, d.detail = "ANALYZE", f"input {h}: {len(finished)}/{len(states)} cells finished"
    return d


def janitor_targets(
    instances: list[VastInstance], manifests: list[Manifest], now: datetime, now_ts: float,
) -> list[tuple[int, str]]:
    """Instances to destroy: past their manifest's deadline, or belonging to no manifest at all.
    Only `soamp-*` labelled instances are ever passed in (VastClient.list_instances)."""
    out: list[tuple[int, str]] = []
    by_id = {iid: m for m in manifests for iid in m.instance_ids}
    known = {f"soamp-{m.experiment}" for m in manifests}
    for i in instances:
        m = by_id.get(i.id)
        if m is not None:
            if m.is_past_deadline(now):
                out.append((i.id, f"past deadline {m.deadline} of {m.experiment}"))
        elif i.label not in known:
            young = i.started_at is not None and now_ts - i.started_at < ORPHAN_GRACE_SECONDS
            if not young:
                out.append((i.id, f"orphan: no manifest for label {i.label!r}"))
    return out
