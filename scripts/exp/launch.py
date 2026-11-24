"""Plan (and, later, launch) an experiment request.

    python scripts/exp/launch.py config/requests/<request>.yaml            # = --plan
    python scripts/exp/launch.py config/requests/<request>.yaml --plan     # no side effects

`--plan` validates the request (including that every organism of the dataset has a
real vector in each requested k-mer/DNABERT-S artifact), expands it to cells,
estimates GPU-hours/cost and checks every cap in config/runs/limits.yaml against the
LIVE instance count (vast.ai) and 24 h spend (W&B ledger). Both lookups are read-only
and free; without credentials (or with --offline) they count as 0 and the output says
so. Exit code 0 = allowed, 1 = refused, 2 = invalid request.

`--go` re-checks every cap against the LIVE instance count (vast.ai) and 24 h spend
(W&B ledger), skips cells that already have a finished W&B run, then rents instances
that bootstrap themselves from W&B artifacts (code bundle + data). Needs env:
WANDB_API_KEY, WANDB_ENTITY, VAST_API_KEY (cloud) -- or a local .env + `vastai set api-key`.
"""
import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

from soamp.runs.budget import estimate, load_cost_model, load_limits, violations
from soamp.runs.bundle import build_tarball, select_files, tracked_files
from soamp.runs.coverage import coverage_problems
from soamp.runs.launch_plan import prebuilt_files
from soamp.runs.launcher import LaunchRefused, launch, plan_usage
from soamp.runs.ledger import OPS_PROJECT, Ledger, WandbLedgerBackend
from soamp.runs.plan import expand, request_hash
from soamp.runs.request import ExperimentRequest
from soamp.runs.vast import VastClient, load_blocked_host_ids
from soamp.utils.config import REPO_ROOT

log = logging.getLogger("launch")


def _request_from_manifest(experiment: str) -> ExperimentRequest:
    """Rebuild the request of the latest launch of `experiment` from the W&B ledger."""
    entity = os.environ.get("WANDB_ENTITY") or None
    rows = [m for m in WandbLedgerBackend(entity).launches() if m.get("experiment") == experiment and m.get("request")]
    if not rows:
        raise LookupError(f"no launch manifest with a stored request for experiment {experiment!r}")
    return ExperimentRequest(**max(rows, key=lambda m: m["created_at"])["request"])


def _git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT).returncode != 0
    return sha + ("-dirty" if dirty else "")


def _publish_bundle_factory(experiment: str, entity: str):
    def publish(extra: list[str]) -> str:
        import wandb

        files = select_files(tracked_files(REPO_ROOT), extra=extra)
        with tempfile.TemporaryDirectory() as d:
            tar = build_tarball(REPO_ROOT, files, Path(d) / "code.tar.gz")
            run = wandb.init(entity=entity, project=OPS_PROJECT, job_type="code_bundle", name=f"bundle-{experiment}")
            art = wandb.Artifact(f"code-{experiment}", type="code")
            art.add_file(str(tar), name="code.tar.gz")
            run.log_artifact(art).wait()
            ref = art.qualified_name
            run.finish()
        log.info("code bundle: %d files -> %s", len(files), ref)
        return ref

    return publish


def _go(req: ExperimentRequest) -> int:
    from soamp.runs.wandb_state import finished_run_names

    entity, key = os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_API_KEY")
    if not entity or not key:
        log.error("WANDB_ENTITY and WANDB_API_KEY must be set")
        return 2
    import wandb

    data_ref = f"{entity}/{OPS_PROJECT}/soamp_{req.dataset}_data:latest"
    try:
        data_art = wandb.Api().artifact(data_ref)
    except Exception as e:
        log.error("data artifact %s not found (%s). Run on the laptop: python scripts/exp/publish_data.py --dataset %s", data_ref, e, req.dataset)
        return 2
    # Prebuilt organism artifacts are never built on the instance: refuse before renting if the
    # published data artifact predates them (the instance would only fail after paying for setup).
    needed = prebuilt_files(req.dataset, req.representations.organism)
    absent = [f for f in needed if f not in data_art.manifest.entries]
    if absent:
        log.error("data artifact %s lacks %s. Re-publish on the laptop: python scripts/exp/publish_data.py --dataset %s", data_ref, absent, req.dataset)
        return 2
    try:
        res = launch(
            req, client=VastClient(blocked_host_ids=load_blocked_host_ids()), ledger=Ledger(WandbLedgerBackend(entity)), limits=load_limits(), cost_model=load_cost_model(),
            finished=finished_run_names(req.experiment, entity), publish_bundle=_publish_bundle_factory(req.experiment, entity),
            data_ref=data_ref, entity=entity, wandb_api_key=key, git_sha=_git_sha(), repo_root=REPO_ROOT,
        )
    except LaunchRefused as e:
        log.error("REFUSED (nothing was rented): %s", e)
        return 1
    if res.skipped_finished:
        log.info("skipped %d cell(s) already finished in W&B", len(res.skipped_finished))
    if not res.launched_cells:
        log.info("nothing to run: every cell of %s is already finished", req.experiment)
        return 0
    log.info("LAUNCHED %s: %d cell(s) on instance(s) %s, estimate $%.2f, deadline %s", req.experiment, len(res.launched_cells), res.instance_ids, res.estimate_usd, res.manifest.deadline)
    log.info("results: W&B project soamp, group %s; status: project soamp-ops. Verify teardown later: vastai show instances", req.experiment)
    return 0


def _plan_usage(offline: bool) -> tuple[int, float, list[str]]:
    if offline:
        return 0, 0.0, ["instance count", "24h spend"]

    def spend() -> float:
        entity = os.environ.get("WANDB_ENTITY")
        if not (entity and os.environ.get("WANDB_API_KEY")):
            raise RuntimeError("WANDB_ENTITY / WANDB_API_KEY not set")
        return Ledger(WandbLedgerBackend(entity)).spend_last_24h()

    return plan_usage(lambda: len(VastClient().list_instances()), spend)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", type=Path, nargs="?")
    ap.add_argument("--retry", metavar="EXPERIMENT", help="re-launch an experiment's unfinished cells from its latest launch manifest in W&B (implies --go)")
    ap.add_argument("--plan", action="store_true", help="validate + estimate only (default)")
    ap.add_argument("--go", action="store_true", help="rent instances and start the run (checks all caps first)")
    ap.add_argument("--offline", action="store_true", help="--plan without the live vast.ai/W&B lookups (counts them as 0)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.retry:
        args.go = True
    elif args.request is None:
        ap.error("a request file (or --retry EXPERIMENT) is required")
    try:
        req = _request_from_manifest(args.retry) if args.retry else ExperimentRequest(**yaml.safe_load(args.request.read_text()))
    except (ValidationError, LookupError) as e:
        log.error("invalid request:\n%s", e)
        return 2
    problems = coverage_problems(req)
    if problems:
        log.error("invalid request:\n  - %s", "\n  - ".join(problems))
        return 2

    if args.go:
        return _go(req)
    cells = expand(req)
    limits, model = load_limits(), load_cost_model()
    running, spend, unchecked = _plan_usage(offline=args.offline)
    # same sharding as --go: instances already running reduce how many this request may use
    est = estimate(cells, model, limits, instances_available=max(limits.max_concurrent_instances - running, 0))
    log.info("experiment %s (request %s): %d cells, dataset=%s, cv=%s", req.experiment, request_hash(req), len(cells), req.dataset, req.cv)
    for arch in sorted({c.arch for c in cells}):
        log.info("  %-28s %d cells", arch, sum(c.arch == arch for c in cells))
    log.info("estimate: %.2f cell-hours on %d instance(s), %.2f h each incl. overhead, $%.2f incl. %.2fx margin (assumed $%.2f/h)",
             est.cell_hours, est.n_instances, est.hours_per_instance, est.cost_usd, limits.safety_margin, model.assumed_price_per_hour_usd)
    log.info("live usage: %d soamp instance(s) running, $%.2f launched in the last 24h%s",
             running, spend, f" ({', '.join(unchecked)} NOT checked, counted as 0)" if unchecked else "")
    problems = violations(est, limits, instances_running=running, spend_last_24h_usd=spend)
    if problems:
        log.error("REFUSED:\n  - %s", "\n  - ".join(problems))
        return 1
    log.info("within all caps%s", f" (except unchecked: {', '.join(unchecked)})" if unchecked else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
