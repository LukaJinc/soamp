"""Run specs: declare which CV experiments to run (and where), as data.

A run spec is a schema-validated YAML (see ``config/vast/``) listing the
``pipeline/train_cv.py`` configs to execute, either explicitly or as a
peptide x organism grid, plus the compute the job wants (GPU, disk, timeout).
Nothing here trains or talks to vast.ai -- it is pure resolution logic, so a
new dataset or representation only means editing a spec, not code.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from soamp.engine.config import TrainConfig
from soamp.utils.config import REPO_ROOT, load_config


class InstanceSpec(BaseModel):
    """Compute requested from vast.ai. Consumed by scripts/vast/launch.sh."""

    model_config = ConfigDict(extra="forbid")

    gpu_name: str = "RTX_4090"
    num_gpus: int = 1
    # Offers are ranked by host reliability (a dropped instance loses the in-flight
    # cell -- there is no resume), not by price. Verified, on-demand hosts only.
    min_reliability: float = 0.99
    min_inet_down_mbps: int = 500
    disk_gb: int = 50
    image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    # Hard wall-clock cap for the whole job; the instance is destroyed after it.
    max_hours: float = 6.0
    # Seconds to wait after a failure/timeout (log upload) before destroying.
    grace_seconds: int = 120


class GridSpec(BaseModel):
    """Experiment configs generated as ``cv_<peptide>_<organism><variant>``."""

    model_config = ConfigDict(extra="forbid")

    peptide: list[str]
    organism: list[str]
    variant: str = ""
    exclude: list[tuple[str, str]] = []
    config_dir: Path = Path("config/train")
    prefix: str = "cv"


class PrepareStep(BaseModel):
    """A command that builds artifacts the experiments need (e.g. PeptideCLM
    features, which want the GPU). Skipped when every `produces` file exists."""

    model_config = ConfigDict(extra="forbid")

    cmd: list[str]
    produces: list[Path]


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Explicit train_cv configs (repo-relative) and/or a generated grid.
    experiments: list[Path] = []
    grid: GridSpec | None = None
    # Files that must exist before any training starts (fail early, not mid-run).
    required_artifacts: list[Path] = []
    # Run in order before any experiment; artifacts they produce need not exist up front.
    prepare: list[PrepareStep] = []
    instance: InstanceSpec = InstanceSpec()

    @model_validator(mode="after")
    def _non_empty(self) -> "RunSpec":
        if not self.experiments and self.grid is None:
            raise ValueError("run spec needs `experiments` and/or `grid`")
        return self


@dataclass(frozen=True)
class Experiment:
    config_path: Path
    exp_id: str
    results_csv: Path  # reports/cv_results_<exp_id>.csv, written by train_cv.py
    wandb_group: str | None = None


def resolve_config_paths(spec: RunSpec, repo_root: Path = REPO_ROOT) -> list[Path]:
    """Explicit configs first, then the grid; de-duplicated, order kept.
    Raises FileNotFoundError listing every missing config."""
    paths = list(spec.experiments)
    if spec.grid is not None:
        g = spec.grid
        excluded = set(g.exclude)
        for p, o in product(g.peptide, g.organism):
            if (p, o) not in excluded:
                paths.append(g.config_dir / f"{g.prefix}_{p}_{o}{g.variant}.yaml")
    unique = list(dict.fromkeys(paths))
    missing = [str(p) for p in unique if not (repo_root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"experiment configs not found: {missing}")
    return unique


def describe_experiment(config_path: Path) -> Experiment:
    cfg = load_config(config_path, TrainConfig)
    return Experiment(
        config_path=config_path,
        exp_id=cfg.exp_id,
        results_csv=cfg.paths.reports_dir / f"cv_results_{cfg.exp_id}.csv",
        wandb_group=cfg.wandb_group,
    )


def pending_experiments(experiments: list[Experiment], finished: frozenset[str] | set[str] = frozenset()) -> list[Experiment]:
    """Skip-if-done: train_cv.py has no resume. A cell is done if its results CSV exists
    locally (written before the wandb upload, so "trained", not "uploaded") or W&B already
    has a *finished* run of that name (`finished`, see soamp.runs.wandb_state) -- the latter
    is what makes a relaunch on a fresh instance idempotent."""
    return [e for e in experiments if not e.results_csv.is_file() and e.exp_id not in finished]


def required_artifacts_missing(spec: RunSpec, repo_root: Path = REPO_ROOT) -> list[Path]:
    produced = {p for st in spec.prepare for p in st.produces}
    return [p for p in spec.required_artifacts if p not in produced and not (repo_root / p).is_file()]


def pending_prepare_steps(spec: RunSpec, repo_root: Path = REPO_ROOT) -> list[PrepareStep]:
    return [st for st in spec.prepare if not all((repo_root / p).is_file() for p in st.produces)]


def offer_query(instance: InstanceSpec) -> str:
    """The `vastai search offers` filter for this spec; sort with `-o 'reliability-'`."""
    return (
        f"gpu_name={instance.gpu_name} num_gpus={instance.num_gpus} "
        f"reliability>{instance.min_reliability} verified=true rentable=true "
        f"disk_space>={instance.disk_gb} inet_down>{instance.min_inet_down_mbps} "
        "direct_port_count>=1 cuda_vers>=12.4"
    )
