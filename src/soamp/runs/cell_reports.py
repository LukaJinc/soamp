"""Instance-side: upload one finished cell's report files to W&B before the instance is gone.

A destroyed vast.ai instance leaves nothing behind. ``pipeline/train_cv.py`` already logs its
metrics and the cv_results table to the training run, but the files it writes under
``reports/`` (train_cv_<exp_id>_log.txt, cv_results_/cv_loss_curves_/cv_snapshots_<exp_id>.csv,
tree-model outputs ...) only lived on the instance. ``scripts/vast/run_experiments.py`` calls
``upload_cell_reports`` after every cell, successful or not, so a failed cell's log survives too.

Uploads go to the ops project (one ``cell_reports`` run + artifact per cell, grouped by
experiment), never to the results project, so they cannot be mistaken for a finished
training run by ``soamp.runs.wandb_state.finished_run_names``.
"""
from __future__ import annotations

from pathlib import Path

from soamp.runs.ledger import OPS_PROJECT


def cell_report_files(reports_dir: Path, exp_id: str) -> list[Path]:
    """Every file under `reports_dir` (recursively) whose name contains `exp_id`, sorted.
    exp_ids end in a config hash, so one cell's id never matches another cell's files."""
    if not exp_id or not reports_dir.is_dir():
        return []
    return sorted(p for p in reports_dir.rglob(f"*{exp_id}*") if p.is_file())


def upload_cell_reports(
    exp_id: str,
    files: list[Path],
    *,
    repo_root: Path,
    group: str | None,
    entity: str | None,
    returncode: int,
    project: str = OPS_PROJECT,
) -> str | None:
    """Log `files` as artifact ``cell_reports-<exp_id>`` (paths kept repo-relative). Returns the
    artifact's qualified name, or None when there is nothing to upload."""
    if not files:
        return None
    import wandb

    run = wandb.init(
        entity=entity, project=project, job_type="cell_reports", name=f"reports-{exp_id}",
        group=group, config={"exp_id": exp_id, "returncode": returncode},
    )
    try:
        art = wandb.Artifact(
            f"cell_reports-{exp_id}", type="cell_reports",
            metadata={"exp_id": exp_id, "returncode": returncode, "files": [str(f.relative_to(repo_root)) for f in files]},
        )
        for f in files:
            art.add_file(str(f), name=str(f.relative_to(repo_root)))
        run.log_artifact(art).wait()
        return art.qualified_name
    finally:
        run.finish()
