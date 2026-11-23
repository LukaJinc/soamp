"""W&B facts for scripts/exp/watch.py (thin IO; the decisions live in soamp.runs.status)."""
from __future__ import annotations

from datetime import datetime, timezone

from soamp.runs.ledger import OPS_PROJECT
from soamp.runs.status import RunInfo
from soamp.runs.wandb_state import PROJECT


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


class WandbStatusBackend:
    def __init__(self, entity: str | None = None):
        import wandb

        self._api = wandb.Api()
        self.entity = entity or self._api.default_entity

    def group_runs(self, experiment: str) -> dict[str, RunInfo]:
        out: dict[str, RunInfo] = {}
        for r in self._api.runs(f"{self.entity}/{PROJECT}", filters={"group": experiment}):
            version = ""
            if r.state == "finished":
                arts = [a.name for a in r.logged_artifacts() if a.name.startswith("cv_results")]
                version = arts[0] if arts else ""
            out[r.name] = RunInfo(
                r.name, r.state, run_id=r.id, artifact_version=version,
                heartbeat_at=_parse(r._attrs.get("heartbeatAt")) or _parse(r.created_at),
            )
        return out

    def status_records(self, experiment: str) -> list[dict]:
        return [
            {k: r.config.get(k) for k in ("rc", "cells", "instance_id", "shard", "finished_at")}
            for r in self._api.runs(f"{self.entity}/{OPS_PROJECT}", filters={"group": experiment, "jobType": "instance_status"})
        ]

    def reports_done(self, experiment: str) -> set[str]:
        """input_hash of every delivered report (analysis or failure) for this experiment."""
        try:
            versions = self._api.artifacts("exp_report", f"{self.entity}/{OPS_PROJECT}/exp_report-{experiment}")
            return {a.metadata.get("input_hash") for a in versions if a.metadata.get("input_hash")}
        except Exception:
            return set()  # no report artifact yet
