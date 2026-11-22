"""`--go`: turn a validated request into rented, self-bootstrapping instances.

Every effect is injected (vast client, ledger, bundle publisher, finished-run
lookup), so the safety logic -- caps re-checked against LIVE instance count and
24 h spend, idempotent skip of finished cells, refuse-before-spend -- is unit
tested without a network. scripts/exp/launch.py wires the real implementations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from soamp.runs.budget import CostModel, Limits, estimate, violations
from soamp.runs.ledger import Ledger, Manifest, deadline_for
from soamp.runs.launch_plan import build_onstart_cmd, build_run_spec, shard, spec_path_for
from soamp.runs.plan import expand, request_hash, write_cell_configs
from soamp.runs.request import ExperimentRequest
from soamp.runs.spec import InstanceSpec
from soamp.runs.vast import LABEL_PREFIX, VastClient

GRACE_SECONDS = 120


class LaunchRefused(Exception):
    """Raised BEFORE anything is rented; message lists every reason."""


@dataclass
class LaunchResult:
    experiment: str
    skipped_finished: list[str] = field(default_factory=list)
    launched_cells: list[str] = field(default_factory=list)
    instance_ids: list[int] = field(default_factory=list)
    estimate_usd: float = 0.0
    manifest: Manifest | None = None


def launch(
    req: ExperimentRequest,
    *,
    client: VastClient,
    ledger: Ledger,
    limits: Limits,
    cost_model: CostModel,
    finished: set[str],
    publish_bundle: Callable[[list[str]], str],  # extra files -> code artifact ref
    data_ref: str,
    entity: str,
    wandb_api_key: str,
    git_sha: str,
    repo_root: Path,
    now: datetime | None = None,
) -> LaunchResult:
    now = now or datetime.now(timezone.utc)
    cells = expand(req)
    todo = [c for c in cells if c.exp_id not in finished]
    result = LaunchResult(req.experiment, skipped_finished=[c.exp_id for c in cells if c.exp_id in finished])
    if not todo:
        return result  # idempotent: everything already finished in W&B

    running = len(client.list_instances())
    spend = ledger.spend_last_24h(now)
    base_instance = InstanceSpec(max_hours=limits.max_hours_per_instance, grace_seconds=GRACE_SECONDS)
    offers = client.search_offers(base_instance, limit=8)
    if not offers:
        raise LaunchRefused("no vast.ai offers match the instance requirements")
    est = estimate(
        todo, cost_model, limits, price_per_hour_usd=offers[0].price_per_hour,
        instances_available=max(limits.max_concurrent_instances - running, 0),
    )
    problems = violations(est, limits, instances_running=running, spend_last_24h_usd=spend)
    if len(offers) < est.n_instances:
        problems.append(f"only {len(offers)} offers for {est.n_instances} instances")
    if problems:
        raise LaunchRefused("; ".join(problems))

    # --- past this point money is spent: materialize files, then rent ---
    shards = shard(todo, est.n_instances)
    max_hours = min(limits.max_hours_per_instance, max(0.5, math.ceil(est.hours_per_instance * 2 * 10) / 10))
    instance = InstanceSpec(max_hours=max_hours, grace_seconds=GRACE_SECONDS)
    extra: list[str] = [str(p) for p in write_cell_configs(req, todo, repo_root)]
    spec_paths: list[Path] = []
    for i, part in enumerate(shards):
        sp = spec_path_for(req.experiment, i)
        (repo_root / sp).parent.mkdir(parents=True, exist_ok=True)
        spec = build_run_spec(req, part, instance)
        (repo_root / sp).write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
        spec_paths.append(sp)
        extra.append(str(sp))
    code_ref = publish_bundle(extra)

    for i, (part, sp) in enumerate(zip(shards, spec_paths)):
        iid = client.create_instance(
            offers[i].id, image=instance.image, disk_gb=instance.disk_gb,
            label=f"{LABEL_PREFIX}{req.experiment}",
            onstart_cmd=build_onstart_cmd(
                code_artifact=code_ref, data_artifact=data_ref, spec_path=str(sp),
                max_hours=instance.max_hours, grace_seconds=instance.grace_seconds,
            ),
            env={
                "WANDB_API_KEY": wandb_api_key, "WANDB_ENTITY": entity, "SOAMP_GIT_SHA": git_sha,
                "SOAMP_EXPERIMENT": req.experiment, "SOAMP_SHARD": str(i),
            },
        )
        result.instance_ids.append(iid)
        result.launched_cells += [c.exp_id for c in part]

    manifest = Manifest(
        experiment=req.experiment, request_hash=request_hash(req), created_at=now.isoformat(),
        deadline=deadline_for(now, max_hours=instance.max_hours, grace_seconds=instance.grace_seconds).isoformat(),
        estimate_usd=est.cost_usd, git_sha=git_sha, cells=[c.exp_id for c in cells], instance_ids=result.instance_ids,
    )
    ledger.record_launch(manifest)
    result.manifest, result.estimate_usd = manifest, est.cost_usd
    return result
