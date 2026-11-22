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
    steps = [s for p in peps for s in ds["prepare"]["peptide"][p]] + [s for o in orgs for s in ds["prepare"]["organism"][o]]
    seen: set[tuple] = set()
    prepare = []
    for cmd, produces in steps:
        if tuple(cmd) not in seen:
            seen.add(tuple(cmd))
            prepare.append({"cmd": cmd, "produces": produces})
    return RunSpec(
        experiments=[cell_config_path(req, c) for c in cells],
        required_artifacts=ds["required_artifacts"],
        prepare=prepare,
        instance=instance,
    )


def build_onstart_cmd(
    *, code_artifact: str, data_artifact: str, spec_path: str, max_hours: float, grace_seconds: int
) -> str:
    """Shell run by vast when the container starts. Secrets (WANDB_API_KEY, ...) come
    from the container env set at create time, never from this string."""
    fetch = (
        "import os, tarfile, wandb; "
        f"d = wandb.Api().artifact({code_artifact!r}).download('/workspace/_code'); "
        f"os.makedirs({WORKDIR!r}, exist_ok=True); "
        "tarfile.open(os.path.join(d, 'code.tar.gz')).extractall("
        f"{WORKDIR!r})"
    )
    steps = [
        f"pip install -q {shlex.quote(WANDB_PIN)}",
        f"python -c {shlex.quote(fetch)}",
        f"cd {WORKDIR}",
        # remote_main.sh downloads $SOAMP_DATA_ARTIFACT, installs the stack, runs the spec,
        # uploads the status record and destroys the instance on exit.
        f"SOAMP_DATA_ARTIFACT={shlex.quote(data_artifact)} setsid nohup bash scripts/vast/remote_main.sh "
        f"{shlex.quote(spec_path)} {max_hours} {grace_seconds} </dev/null >/workspace/onstart.log 2>&1 &",
    ]
    return " && ".join(steps)


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
    return list(dict.fromkeys(out))
