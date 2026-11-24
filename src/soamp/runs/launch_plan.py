"""Pure helpers for `--go`: shard cells over instances, build the on-instance
RunSpec, and build the self-bootstrapping onstart command.

The cloud caller cannot SSH to a vast host, so the instance fetches everything
itself from W&B artifacts (code bundle + data) with the pinned wandb version,
then hands over to ``scripts/vast/remote_main.sh`` (which owns the timeout and
the destroy-on-exit trap).
"""
from __future__ import annotations

import shlex
from pathlib import Path

from soamp.runs.plan import Cell, cell_config_path
from soamp.runs.request import DATASETS, ExperimentRequest
from soamp.runs.spec import InstanceSpec, RunSpec

# Same pin as pyproject.toml: newer wandb authenticates against a host the cloud proxy blocks.
WANDB_PIN = "wandb==0.28.2"
WORKDIR = "/workspace/soamp"


def shard(cells: list[Cell], n: int) -> list[list[Cell]]:
    """Round-robin so every instance gets a similar mix; empty shards dropped."""
    if n < 1:
        raise ValueError("n must be >= 1")
    out = [cells[i::n] for i in range(n)]
    return [s for s in out if s]


def build_run_spec(req: ExperimentRequest, cells: list[Cell], instance: InstanceSpec) -> RunSpec:
    """The RunSpec executed by scripts/vast/run_experiments.py on the instance."""
    ds = DATASETS[req.dataset]
    peps = sorted({c.peptide for c in cells})
    orgs = sorted({c.organism for c in cells})
    steps = [s for p in peps for s in ds["prepare"]["peptide"][p]] + [s for o in orgs for s in ds["prepare"]["organism"].get(o, [])]
    seen: set[tuple] = set()
    prepare = []
    for cmd, produces in steps:
        if tuple(cmd) not in seen:
            seen.add(tuple(cmd))
            prepare.append({"cmd": cmd, "produces": produces})
    return RunSpec(
        experiments=[cell_config_path(req, c) for c in cells],
        required_artifacts=list(ds["required_artifacts"]) + prebuilt_files(req.dataset, orgs),
        prepare=prepare,
        instance=instance,
    )


_ONSTART = r"""
set -u
log() { echo "[onstart $(date -u +%H:%M:%S)] $*"; }
for v in CONTAINER_ID CONTAINER_API_KEY; do
  if [ -z "${!v:-}" ]; then
    val=$(tr '\0' '\n' </proc/1/environ | sed -n "s/^$v=//p" | head -1)
    [ -n "$val" ] && export "$v=$val"
  fi
done
# Nothing below may leave a rented-but-idle instance behind: any failure destroys it.
die() {
  log "FATAL: $*; destroying instance ${CONTAINER_ID:-?}"
  pip install -q vastai >/dev/null 2>&1
  yes | vastai destroy instance "${CONTAINER_ID:-}" --api-key "${CONTAINER_API_KEY:-}"
  exit 1
}
# Some hosts are blocked by W&B (api.wandb.ai answers 403 to everything): fail fast, cheaply.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://api.wandb.ai/ || true)
case "$code" in 403|000|"") die "api.wandb.ai unreachable from this host (http '$code')";; esac
timeout 900 pip install -q '__WANDB_PIN__' || die "pip install wandb failed"
timeout 900 python - <<'PY' || die "code bundle download failed"
import os, tarfile, wandb
d = wandb.Api().artifact(__CODE_REF__).download('/workspace/_code')
os.makedirs(__WORKDIR_PY__, exist_ok=True)
tarfile.open(os.path.join(d, 'code.tar.gz')).extractall(__WORKDIR_PY__)
PY
cd __WORKDIR__ || die "code bundle missing"
# remote_main.sh owns the job timeout and its own destroy-on-exit trap from here on.
SOAMP_DATA_ARTIFACT=__DATA_REF_Q__ setsid nohup bash scripts/vast/remote_main.sh __SPEC_Q__ __MAX_HOURS__ __GRACE__ </dev/null >/workspace/onstart.log 2>&1 &
log "handed over to remote_main.sh"
"""


def build_onstart_cmd(
    *, code_artifact: str, data_artifact: str, spec_path: str, max_hours: float, grace_seconds: int
) -> str:
    """Script vast runs when the container starts. Secrets (WANDB_API_KEY, ...) come from the
    container env set at create time, never from this text. Every early failure destroys the
    instance (see `die`), so a broken bootstrap cannot leave an idle instance billing."""
    return (
        _ONSTART.replace("__WANDB_PIN__", WANDB_PIN)
        .replace("__CODE_REF__", repr(code_artifact))
        .replace("__WORKDIR_PY__", repr(WORKDIR))
        .replace("__WORKDIR__", WORKDIR)
        .replace("__DATA_REF_Q__", shlex.quote(data_artifact))
        .replace("__SPEC_Q__", shlex.quote(spec_path))
        .replace("__MAX_HOURS__", str(max_hours))
        .replace("__GRACE__", str(grace_seconds))
    )


def spec_path_for(experiment: str, shard_index: int) -> Path:
    return Path("config/vast/generated") / f"{experiment}_{shard_index}.yaml"


def data_files(dataset: str) -> list[str]:
    """Repo-relative files the data artifact must contain: required inputs + everything the
    prepare steps would produce (shipping them avoids rebuilding features on the instance)."""
    from soamp.runs.request import DATASETS

    ds = DATASETS[dataset]
    out = list(ds["required_artifacts"])
    for group in ds["prepare"].values():
        for steps in group.values():
            for _cmd, produces in steps:
                out.extend(produces)
    for group in ds.get("prebuilt", {}).values():
        for files in group.values():
            out.extend(files)
    return list(dict.fromkeys(out))


def prebuilt_files(dataset: str, organisms: list[str] | set[str]) -> list[str]:
    """Repo-relative prebuilt organism artifacts (never built on the instance) these
    organism representations need; they must already be in the data artifact."""
    pre = DATASETS[dataset].get("prebuilt", {}).get("organism", {})
    return [f for o in sorted(organisms) for f in pre.get(o, [])]
