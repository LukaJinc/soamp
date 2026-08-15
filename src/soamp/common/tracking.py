"""Experiment-tracking adapters. Business/orchestration code calls this
interface (log_config/log_scalar/log_metrics/log_artifact/close), never
touches a backend's SDK directly -- LocalFileTracker and WandbTracker both
implement the same ExperimentTracker Protocol, so any pipeline script
picks one via config.tracking.backend + build_tracker() with zero changes
to its own logic. LocalFileTracker's JSON-lines can also be replayed into
wandb after the fact if a run was only ever logged locally.

Lives in soamp.common (not soamp.engine) because every pipeline stage --
curation, labeling, dataset assembly, features, and the training engine --
uses this, not just training; soamp.engine is specifically the training
loop/checkpointing/eval-loop package. See soamp.common.thresholds for the
same "shared module, not owned by any one stage" precedent.

log_artifact gives inter-stage lineage per CLAUDE.md sec 3: a run logs the
dataset(s) it consumed and the artifact(s) it produced, with depends_on
pointing at its parent artifacts -- not just scalar metrics. Each call
returns a handle (backend-specific) that a later log_artifact call in the
SAME process/run can pass back via depends_on to link the two directly;
a later call in a DIFFERENT process (the common case across this
multi-script pipeline, where each stage is a separate `python
pipeline/...py` invocation) instead passes the parent's artifact *name* as
a bare string -- WandbTracker resolves a bare string via
`use_artifact("name:latest")` since wandb requires an explicit
version/alias suffix to resolve a cross-run reference reliably.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import wandb
from pydantic import BaseModel, ConfigDict, model_validator


class ExperimentTracker(Protocol):
    def log_config(self, config: dict[str, Any]) -> None: ...
    def log_scalar(self, name: str, value: float, step: int) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        paths: list[Path],
        metadata: dict[str, Any] | None = None,
        depends_on: list[Any] | None = None,
    ) -> Any: ...
    def close(self) -> None: ...


class TrackingBackendError(ValueError):
    """Raised when an unknown tracking backend is requested."""


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exp_id: str = "baseline_mlp_v1"
    backend: str = "local"
    wandb_project: str = "soamp"

    @model_validator(mode="after")
    def _backend_must_be_known(self) -> "TrackingConfig":
        if self.backend not in ("local", "wandb"):
            raise ValueError(f"backend must be 'local' or 'wandb', got {self.backend!r}")
        return self


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    return [path]


class LocalFileTracker:
    """Writes <run_dir>/<exp_id>/metrics.jsonl (one JSON object per
    log_scalar call: {"step","name","value","ts"}),
    <run_dir>/<exp_id>/run_metadata.json (resolved config dump, written by
    log_config, called once at run start), and
    <run_dir>/<exp_id>/artifacts/<name>.json (one manifest per
    log_artifact call: file paths + sha256 hashes + metadata + resolved
    depends_on names -- a content-hash-based versioning proxy, since there's
    no backend to assign a real version number)."""

    def __init__(self, run_dir: Path, exp_id: str) -> None:
        self.run_dir = Path(run_dir) / exp_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"
        self.artifacts_dir = self.run_dir / "artifacts"

    def log_config(self, config: dict[str, Any]) -> None:
        with open(self.metadata_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
            f.write("\n")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        record = {"step": step, "name": name, "value": value, "ts": time.time()}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        paths: list[Path],
        metadata: dict[str, Any] | None = None,
        depends_on: list[Any] | None = None,
    ) -> dict[str, Any]:
        files = []
        for p in paths:
            for f in _iter_files(p):
                files.append({"path": str(f), "sha256": _hash_file(f), "bytes": f.stat().st_size})
        manifest = {
            "name": name,
            "type": artifact_type,
            "metadata": metadata or {},
            "depends_on": [d["name"] if isinstance(d, dict) else str(d) for d in (depends_on or [])],
            "files": files,
            "logged_at": time.time(),
        }
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(self.artifacts_dir / f"{name}.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def close(self) -> None:
        pass


class WandbTracker:
    """wandb-backed ExperimentTracker. Reads the API key from the
    WANDB_API_KEY env var (loaded from .env by the orchestration script,
    never hardcoded or committed -- see .env.example). `mode="disabled"`
    makes wandb.init() a local no-op with no network calls or valid key
    required, used by this module's own tests."""

    def __init__(
        self,
        project: str,
        exp_id: str,
        entity: str | None = None,
        mode: str | None = None,
    ) -> None:
        # mode=None defers to WANDB_MODE if set (e.g. "disabled" in tests),
        # else wandb's own default ("online") -- an explicit mode= always wins.
        mode = mode or os.environ.get("WANDB_MODE", "online")
        self._run = wandb.init(project=project, entity=entity, name=exp_id, mode=mode)

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._run.log({name: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        paths: list[Path],
        metadata: dict[str, Any] | None = None,
        depends_on: list[Any] | None = None,
    ) -> wandb.Artifact:
        for dep in depends_on or []:
            if isinstance(dep, wandb.Artifact):
                self._run.use_artifact(dep)
            else:
                dep_str = str(dep)
                self._run.use_artifact(dep_str if ":" in dep_str else f"{dep_str}:latest")
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
        for p in paths:
            p = Path(p)
            if p.is_dir():
                artifact.add_dir(str(p))
            else:
                artifact.add_file(str(p))
        self._run.log_artifact(artifact)
        return artifact

    def close(self) -> None:
        self._run.finish()


def build_tracker(tracking_cfg: TrackingConfig, local_run_dir: Path) -> ExperimentTracker:
    """Dispatches on tracking_cfg.backend to construct the configured
    ExperimentTracker. local_run_dir is only used by the local backend
    (ignored for wandb) -- every orchestration script across
    pipeline/{curation,labeling,data,features,train} calls this
    identically instead of re-deriving the if/elif dispatch itself."""
    if tracking_cfg.backend == "local":
        return LocalFileTracker(local_run_dir, tracking_cfg.exp_id)
    if tracking_cfg.backend == "wandb":
        return WandbTracker(
            project=tracking_cfg.wandb_project,
            exp_id=tracking_cfg.exp_id,
            entity=os.environ.get("WANDB_ENTITY") or None,
        )
    raise TrackingBackendError(f"unknown tracking backend: {tracking_cfg.backend!r}")
