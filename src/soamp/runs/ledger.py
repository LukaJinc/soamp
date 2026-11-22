"""Launch manifests and the spend ledger, durable in W&B (project ``soamp-ops``).

Cloud VMs are ephemeral, so W&B is the only state that survives between a
launch session, the hourly watcher and an analysis session. One W&B run per
launch (job_type ``launch``) carries the manifest in its config and the
estimated cost in its summary; the same manifest is also logged as the artifact
``exp_manifest-<experiment>``. Spend in the last 24 h is the sum of those
estimates -- reconciled against real run times at analysis time.

The W&B calls sit behind ``LedgerBackend`` so everything else is pure and tested.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

OPS_PROJECT = "soamp-ops"
WINDOW = timedelta(hours=24)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


@dataclass
class Manifest:
    experiment: str
    request_hash: str
    created_at: str  # ISO 8601 UTC
    deadline: str  # after this, unfinished cells are `lost` and the janitor may destroy instances
    estimate_usd: float
    git_sha: str
    cells: list[str]  # expected W&B run names (exp_ids)
    instance_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Manifest":
        return Manifest(**d)

    def is_past_deadline(self, now: datetime) -> bool:
        return now >= _utc(self.deadline)


def deadline_for(now: datetime, *, max_hours: float, grace_seconds: int, buffer_minutes: float = 30.0) -> datetime:
    """Instance wall-clock cap + destroy grace + slack for boot/bootstrap."""
    return now + timedelta(hours=max_hours, seconds=grace_seconds, minutes=buffer_minutes)


def spend_in_window(launches: list[dict], now: datetime) -> float:
    """Sum `estimate_usd` of launches created within the last 24 h."""
    return sum(
        float(m["estimate_usd"]) for m in launches if now - _utc(m["created_at"]) <= WINDOW
    )


class LedgerBackend(Protocol):
    def log_launch(self, manifest: Manifest) -> None: ...
    def launches(self) -> list[dict]: ...  # manifest dicts, any age


class Ledger:
    def __init__(self, backend: LedgerBackend):
        self._b = backend

    def record_launch(self, manifest: Manifest) -> None:
        self._b.log_launch(manifest)

    def spend_last_24h(self, now: datetime | None = None) -> float:
        return spend_in_window(self._b.launches(), now or datetime.now(timezone.utc))

    def open_manifests(self, now: datetime | None = None) -> list[Manifest]:
        """Launches still inside their window (a later phase adds explicit closing)."""
        t = now or datetime.now(timezone.utc)
        return [Manifest.from_dict(d) for d in self._b.launches() if not Manifest.from_dict(d).is_past_deadline(t)]


class WandbLedgerBackend:
    """Real backend. Imports wandb lazily so the module stays importable without it."""

    def __init__(self, entity: str | None = None, project: str = OPS_PROJECT):
        self.entity, self.project = entity, project

    def log_launch(self, manifest: Manifest) -> None:
        import wandb

        run = wandb.init(
            entity=self.entity, project=self.project, job_type="launch",
            name=f"launch-{manifest.experiment}-{manifest.created_at[:19]}",
            group=manifest.experiment, config=manifest.to_dict(),
        )
        run.summary["estimate_usd"] = manifest.estimate_usd
        art = wandb.Artifact(f"exp_manifest-{manifest.experiment}", type="manifest")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(manifest.to_dict(), indent=2))
            art.add_file(str(p))
            run.log_artifact(art).wait()
        run.finish()

    def launches(self) -> list[dict]:
        import wandb

        api = wandb.Api()
        path = f"{self.entity}/{self.project}" if self.entity else self.project
        keys = Manifest.__dataclass_fields__.keys()
        return [
            {k: r.config[k] for k in keys if k in r.config}
            for r in api.runs(path, filters={"jobType": "launch"})
            if all(k in r.config for k in ("created_at", "estimate_usd", "deadline"))
        ]
