"""W&B queries about experiment cells (lazy wandb import; the seam tests fake)."""
from __future__ import annotations

import os

PROJECT = "soamp"


def finished_run_names(group: str, entity: str | None = None, project: str = PROJECT) -> set[str]:
    """Names of runs in `group` whose state is `finished`."""
    import wandb

    ent = entity or os.environ.get("WANDB_ENTITY")
    path = f"{ent}/{project}" if ent else project
    return {r.name for r in wandb.Api().runs(path, filters={"group": group}) if r.state == "finished"}
