"""Expand an ExperimentRequest into concrete, deterministic experiment cells.

A *cell* is one ``pipeline/train_cv.py`` run: one architecture hyperparameter
combination x peptide representation x organism representation x seed. Each
cell becomes a small ``_base_``-composed YAML under
``config/train/generated/<experiment>/`` that validates against ``TrainConfig``.

Determinism is the point: the W&B run name (``exp_id``) embeds a hash of the
resolved config, so re-expanding the same request gives the same names, and a
relaunch can skip cells that already have a finished W&B run.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from soamp.runs.request import (
    ARCHITECTURES,
    DATASETS,
    ArchitectureRequest,
    ExperimentRequest,
)
from soamp.utils.config import REPO_ROOT

GENERATED_DIR = Path("config/train/generated")


@dataclass(frozen=True)
class Cell:
    exp_id: str  # also the W&B run name and train_cv.py's exp_id
    arch: str
    peptide: str
    organism: str
    seed: int
    epochs: int | None
    config: dict  # YAML body without `_base_`


def request_hash(req: ExperimentRequest) -> str:
    """Stable 8-char id of the request's content (same request -> same id)."""
    return _sha(req.model_dump(mode="json"))[:8]


def _sha(obj: object) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _combos(axes: dict[str, list]) -> list[dict]:
    names = list(axes)
    return [dict(zip(names, vals)) for vals in product(*(axes[n] for n in names))]


def _arch_combos(a: ArchitectureRequest) -> list[dict]:
    """Cartesian product over this architecture's own axes only (never across
    architectures): shared optimiser fields + architecture-specific fields + epochs."""
    axes: dict[str, list] = {}
    for name, v in a.common().model_dump().items():
        if v is not None:
            axes[name] = list(v) if isinstance(v, list) else [v]
    axes.update(a.specific().axes())
    if a.epochs is not None:
        axes["epochs"] = list(a.epochs) if isinstance(a.epochs, list) else [a.epochs]
    return _combos(axes)


def organism_slug(organism: str | None) -> str:
    """'all' for the pooled cells, else 'e_coli' for 'Escherichia coli ATCC 25922'."""
    if organism is None:
        return "all"
    genus, species = organism.split()[:2]
    return f"{genus[0]}_{species}".lower()


def _cell_body(req: ExperimentRequest, arch: str, combo: dict, pep: str, org: str, seed: int, rhash: str,
               organism_filter: str | None = None) -> dict:
    ds = DATASETS[req.dataset]
    feat, scaler, pep_method = ds["peptide"][pep]
    vocab, org_method = ds["organism"][org]
    spec = ARCHITECTURES[arch]
    arch_fields = {k: v for k, v in combo.items() if k in spec.hparams_cls.model_fields}
    body: dict = {
        "input_files": {
            "peptide_features_filename": feat,
            "peptide_feature_scaler_filename": scaler,
            "organism_vocab_filename": vocab,
        },
        "model": {"architecture": arch, **({arch: arch_fields} if arch_fields else {})},
        "loop": {"seed": seed},
        "cv": {"snapshot_every_n_epochs": req.snapshot_every_n_epochs},
        "wandb_group": req.experiment,
        "wandb_tags": [f"{req.dataset}_dataset", pep_method, org_method, arch, "cv", f"req_{rhash}",
                       f"organism_{organism_slug(organism_filter)}"],
    }
    if req.class_balancing != "auto":
        body["class_balancing"] = {"mode": req.class_balancing}
    if organism_filter is not None:
        body["data_filter"] = {"organism": organism_filter}
    optim = {k: combo[k] for k in ("learning_rate", "weight_decay") if k in combo and spec.uses_common}
    if optim:
        body["optim"] = optim
    for k in ("epochs", "batch_size"):
        if k in combo and spec.uses_common:
            body["loop"][k] = combo[k]
    return body


def expand(req: ExperimentRequest) -> list[Cell]:
    """All cells for a request, de-duplicated and in a stable order."""
    rhash = request_hash(req)
    cells: dict[str, Cell] = {}
    for a in req.architectures:
        tag = ARCHITECTURES[a.name].tag
        for combo, pep, org, seed, subset in product(
            _arch_combos(a), req.representations.peptide, req.representations.organism, req.seeds, req.organism_filters()
        ):
            body = _cell_body(req, a.name, combo, pep, org, seed, rhash, subset)
            exp_id = f"{req.experiment}__{tag}__{pep}_{org}__{organism_slug(subset)}__{_sha(body)[:6]}"
            cells.setdefault(
                exp_id,
                Cell(exp_id, a.name, pep, org, seed, combo.get("epochs"), {"exp_id": exp_id, **body}),
            )
    out = sorted(cells.values(), key=lambda c: c.exp_id)
    if req.sampling.mode == "random" and req.sampling.n < len(out):
        out = sorted(random.Random(rhash).sample(out, req.sampling.n), key=lambda c: c.exp_id)
    return out


def cell_config_path(req: ExperimentRequest, cell: Cell) -> Path:
    return GENERATED_DIR / req.experiment / f"{cell.exp_id}.yaml"


def write_cell_configs(req: ExperimentRequest, cells: list[Cell], repo_root: Path = REPO_ROOT) -> list[Path]:
    """Write one `_base_`-composed YAML per cell; returns repo-relative paths.
    The base is the dataset's config (data dir, thresholds, unit), resolved relative
    to the generated file (``config/train/generated/<experiment>/`` -> ``../../``)."""
    base = f"../../{DATASETS[req.dataset]['base_config']}"
    paths: list[Path] = []
    for c in cells:
        rel = cell_config_path(req, c)
        target = repo_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump({"_base_": base, **c.config}, sort_keys=False))
        paths.append(rel)
    return paths
