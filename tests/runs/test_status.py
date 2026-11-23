from datetime import datetime, timedelta, timezone

from soamp.runs.ledger import Manifest
from soamp.runs.status import (
    RunInfo,
    classify_cell,
    decide,
    experiment_state,
    input_hash,
    janitor_targets,
)
from soamp.runs.vast import VastInstance

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
FRESH = NOW - timedelta(minutes=2)


def _m(cells=("a", "b"), deadline_in_h=2.0, request=True, iids=(7,)):
    return Manifest("exp", "h", (NOW - timedelta(hours=1)).isoformat(), (NOW + timedelta(hours=deadline_in_h)).isoformat(),
                    1.0, "sha", list(cells), list(iids), {"experiment": "exp"} if request else {})


def _run(name, state="finished", hb=FRESH, ver="v1"):
    return RunInfo(name, state, run_id=f"id-{name}", heartbeat_at=hb, artifact_version=ver)


def _cls(run=None, records=(), alive=True, past=False):
    return classify_cell("a", run, status_records=list(records), instance_alive=alive, past_deadline=past, now=NOW)


def test_classify_cell_all_states():
    assert _cls(_run("a")) == "finished"
    assert _cls(_run("a", "crashed")) == "crashed"
    assert _cls(_run("a", "running")) == "running"
    assert _cls(_run("a", "running", hb=NOW - timedelta(minutes=30))) == "stale"
    assert _cls(_run("a", "running"), past=True) == "stale"
    assert _cls() == "missing"  # still booting
    assert _cls(past=True) == "lost"
    assert _cls(records=[{"cells": {}}], alive=False) == "lost"  # instance ended without it
    assert _cls(records=[{"cells": {"a": {"returncode": 1}}}], alive=False) == "crashed"
    assert _cls(records=[{"cells": {"a": {"returncode": 0}}}], alive=True) == "missing"


def test_experiment_state():
    assert experiment_state({"a": "finished", "b": "running"}) == "running"
    assert experiment_state({"a": "finished", "b": "missing"}) == "running"
    assert experiment_state({"a": "finished", "b": "finished"}) == "complete"
    assert experiment_state({"a": "finished", "b": "crashed"}) == "partial"
    assert experiment_state({"a": "lost", "b": "stale"}) == "failed"


def _decide(runs, records=(), instances=(), launches=None, done=frozenset(), manifest=None):
    return decide(manifest or _m(), runs=runs, status_records=list(records), instances=list(instances),
                  launches=launches if launches is not None else [{"experiment": "exp"}], reports_done=set(done), now=NOW)


def test_running_experiment_takes_no_action():
    d = _decide({"a": _run("a")}, instances=[VastInstance(7, "soamp-exp", "running", 0.5)])
    assert d.state == "running" and d.action == "NONE"


def test_complete_experiment_is_analyzed_once():
    runs = {"a": _run("a"), "b": _run("b")}
    d = _decide(runs)
    assert d.state == "complete" and d.action == "ANALYZE"
    h = input_hash(list(runs.values()))
    assert d.detail.startswith(f"input {h}")
    again = _decide(runs, done={h})
    assert again.action == "NONE" and "already analyzed" in again.detail
    # a new artifact version changes the hash -> analyzed again
    changed = {"a": _run("a", ver="v2"), "b": _run("b")}
    assert _decide(changed, done={h}).action == "ANALYZE"


def test_partial_with_a_training_crash_is_analyzed_not_retried():
    d = _decide({"a": _run("a"), "b": _run("b", "crashed")}, records=[{"rc": 1, "cells": {"b": {"returncode": 1}}}])
    assert d.state == "partial" and d.action == "ANALYZE" and not d.retry_cells


def test_infra_failure_is_retried_once_then_reported():
    # instance died at setup (rc=10), no cell ever ran
    rec = [{"rc": 10, "cells": {}}]
    d = _decide({}, records=rec, manifest=_m(deadline_in_h=-1))
    assert d.action == "RETRY" and sorted(d.retry_cells) == ["a", "b"]
    exhausted = _decide({}, records=rec, manifest=_m(deadline_in_h=-1), launches=[{"experiment": "exp"}] * 2)
    assert exhausted.action == "REPORT_FAILURE"
    assert _decide({}, manifest=_m(deadline_in_h=-1, request=False)).action == "REPORT_FAILURE"  # nothing to rebuild from


def test_vanished_instance_with_no_record_is_infra_failure():
    assert _decide({}, manifest=_m(deadline_in_h=-1)).action == "RETRY"


def test_all_cells_crashed_is_a_failure_report_without_retry():
    d = _decide({"a": _run("a", "crashed"), "b": _run("b", "crashed")})
    assert d.state == "failed" and d.action == "REPORT_FAILURE"
    m = _m()
    from soamp.runs.status import failure_key
    again = _decide({"a": _run("a", "crashed"), "b": _run("b", "crashed")}, done={failure_key(m)}, manifest=m)
    assert again.action == "NONE" and "already reported" in again.detail  # announced once, not every tick


def _inst(i, label, age_s=3600.0):
    return VastInstance(i, label, "running", 0.5, started_at=NOW.timestamp() - age_s)


def test_janitor_destroys_overdue_and_old_orphans_only():
    ms = [_m(deadline_in_h=-1, iids=(1,)), Manifest("live", "h", NOW.isoformat(), (NOW + timedelta(hours=1)).isoformat(), 1, "s", ["c"], [2])]
    ins = [_inst(1, "soamp-exp"), _inst(2, "soamp-live"), _inst(3, "soamp-ghost"), _inst(4, "soamp-new", age_s=60)]
    got = dict(janitor_targets(ins, ms, NOW, NOW.timestamp()))
    assert set(got) == {1, 3}
    assert "deadline" in got[1] and "orphan" in got[3]
