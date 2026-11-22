"""Cost/size caps for unattended launches -- pure logic, no network.

The launcher gathers the two live inputs (instances currently running, spend in
the last 24 h) from vast.ai / the W&B ledger and passes them in; everything
else is deterministic and unit-tested. Limits live in ``config/runs/limits.yaml``
(committed). Callers may *lower* a limit (``Limits.lowered``) but never raise it:
raising means editing the committed file, i.e. a reviewed change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soamp.runs.plan import Cell
from soamp.utils.config import REPO_ROOT

LIMITS_PATH = Path("config/runs/limits.yaml")
COST_MODEL_PATH = Path("config/runs/cost_model.yaml")


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cost_per_request_usd: float = Field(gt=0)
    max_concurrent_instances: int = Field(ge=1)
    max_cost_per_24h_usd: float = Field(gt=0)
    max_cells: int = Field(ge=1)
    max_hours_per_instance: float = Field(gt=0)
    # Estimates are optimistic by nature (host speed varies, no resume on failure).
    safety_margin: float = Field(ge=1.0)

    def lowered(self, **overrides: float) -> "Limits":
        """Return a copy with some limits tightened. Raising a limit raises."""
        data = self.model_dump()
        for k, v in overrides.items():
            if k not in data:
                raise KeyError(f"unknown limit {k!r}")
            if k == "safety_margin":
                ok = v >= data[k]  # a larger margin is the tighter setting
            else:
                ok = v <= data[k]
            if not ok:
                raise ValueError(f"limit {k}={v} would loosen the committed value {data[k]}")
            data[k] = v
        return Limits(**data)


class CostModel(BaseModel):
    """Seed timings (GPU). Replace with medians of past W&B runs once there is history."""

    model_config = ConfigDict(extra="forbid")

    # wall-clock minutes for one 5-fold CV cell of 100 epochs, by peptide representation
    minutes_per_100_epochs: dict[str, float]
    default_epochs: int = 30  # TrainConfig.loop.epochs default
    # per instance: boot, uv/torch install, data download, feature build
    instance_overhead_minutes: float = 15.0
    assumed_price_per_hour_usd: float = 0.8


def load_limits(path: Path = LIMITS_PATH, repo_root: Path = REPO_ROOT) -> Limits:
    return Limits(**yaml.safe_load((repo_root / path).read_text()))


def load_cost_model(path: Path = COST_MODEL_PATH, repo_root: Path = REPO_ROOT) -> CostModel:
    return CostModel(**yaml.safe_load((repo_root / path).read_text()))


@dataclass(frozen=True)
class Estimate:
    n_cells: int
    n_instances: int
    cell_hours: float  # pure training time summed over cells
    hours_per_instance: float  # incl. overhead, the busiest instance
    cost_usd: float  # incl. safety margin


def cell_hours(cell: Cell, model: CostModel) -> float:
    if cell.peptide not in model.minutes_per_100_epochs:
        raise KeyError(f"no timing for peptide representation {cell.peptide!r} in cost model")
    epochs = cell.epochs or model.default_epochs
    return model.minutes_per_100_epochs[cell.peptide] * epochs / 100 / 60


def estimate(
    cells: list[Cell],
    model: CostModel,
    limits: Limits,
    *,
    price_per_hour_usd: float | None = None,
    instances_available: int | None = None,
) -> Estimate:
    """Shard cells over the fewest instances that keep each under
    ``max_hours_per_instance``, never more than ``instances_available``."""
    price = price_per_hour_usd if price_per_hour_usd is not None else model.assumed_price_per_hour_usd
    avail = limits.max_concurrent_instances if instances_available is None else instances_available
    total = sum(cell_hours(c, model) for c in cells)
    overhead = model.instance_overhead_minutes / 60
    n = 1
    while n < max(avail, 1) and total / n + overhead > limits.max_hours_per_instance:
        n += 1
    hours_each = total / n + overhead
    cost = (total + overhead * n) * price * limits.safety_margin
    return Estimate(len(cells), n, total, hours_each, cost)


def violations(
    est: Estimate,
    limits: Limits,
    *,
    instances_running: int,
    spend_last_24h_usd: float,
) -> list[str]:
    """Every reason this launch must be refused (empty list = allowed)."""
    out: list[str] = []
    if est.n_cells == 0:
        out.append("request expands to 0 cells")
    if est.n_cells > limits.max_cells:
        out.append(f"{est.n_cells} cells > max_cells {limits.max_cells}")
    if est.hours_per_instance > limits.max_hours_per_instance:
        out.append(
            f"needs {est.hours_per_instance:.2f} h per instance > max_hours_per_instance "
            f"{limits.max_hours_per_instance} even when sharded over {est.n_instances} instance(s)"
        )
    if instances_running + est.n_instances > limits.max_concurrent_instances:
        out.append(
            f"{instances_running} running + {est.n_instances} requested > "
            f"max_concurrent_instances {limits.max_concurrent_instances}"
        )
    if est.cost_usd > limits.max_cost_per_request_usd:
        out.append(f"estimated ${est.cost_usd:.2f} > max_cost_per_request_usd ${limits.max_cost_per_request_usd:.2f}")
    if spend_last_24h_usd + est.cost_usd > limits.max_cost_per_24h_usd:
        out.append(
            f"${spend_last_24h_usd:.2f} spent in last 24h + ${est.cost_usd:.2f} > "
            f"max_cost_per_24h_usd ${limits.max_cost_per_24h_usd:.2f}"
        )
    return out
