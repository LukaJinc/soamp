"""High-level experiment request: what the user asks for in natural language,
as schema-validated data.

A request names one experiment (= one W&B group) and lists architectures, each
with its OWN hyperparameter search space, crossed with peptide x organism
representations. Cross-validation is the default. Nothing here trains, expands
or launches anything -- ``soamp.runs.plan`` expands a request into cells and
``soamp.runs.budget`` decides whether it may run.

Search-space convention (per architecture, validated against that architecture's
own hyperparameter model, so a key from another architecture raises):
  * a scalar is a fixed value;
  * a list of scalars is a grid axis (``learning_rate: [1e-3, 1e-4]``);
  * list-typed hyperparameters (``hidden_dims``) take a flat list of ints as ONE
    value, and a list of lists as a grid axis.
"""
from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Dataset -> which representation methods have artifacts/configs for it. Adding a
# dataset or a method here is the only place that needs to change (plan.py reads it).
DATASETS: dict[str, dict] = {
    "dev": {
        # config the generated cells inherit (data dir, thresholds, unit) via `_base_`
        "base_config": "cv_dev_peptideclm_vocab.yaml",
        "peptide": {
            "rdkit": ("peptide_features_rdkit.csv", "peptide_feature_scaler_rdkit.json", "rdkit_descriptors"),
            "peptideclm": ("peptide_features_peptideclm.csv", "peptide_feature_scaler_peptideclm.json", "peptideclm_embedding"),
            "molgraph": ("peptide_features_molgraph.csv", "peptide_feature_scaler_molgraph.json", "molecular_graph"),
        },
        # only the vocab organism artifact exists for the dev dataset so far
        "organism": {
            "vocab": ("organism_vocab_vocab_embedding.json", "vocab_embedding"),
        },
        # Inputs every cell needs (shipped in the W&B data artifact, never built on the instance).
        "required_artifacts": ["data/dev/mic_classification_dataset.csv", "data/dev/peptide_split.csv"],
        # Steps that build a representation's artifacts if they are not already in the data
        # artifact (skipped when every `produces` file exists). Mirrors config/vast/dev_sweep_v1.yaml.
        "prepare": {
            "peptide": {
                "rdkit": [
                    (["python", "pipeline/features/01_build_peptide_features.py", "--config", "config/features/dev_peptide_rdkit.yaml"], ["data/dev/peptide_features_rdkit.csv"]),
                    (["python", "pipeline/features/03_fit_peptide_scaler.py", "--config", "config/features/dev_peptide_rdkit.yaml"], ["data/dev/peptide_feature_scaler_rdkit.json"]),
                ],
                "peptideclm": [
                    (["python", "pipeline/features/01_build_peptide_features.py", "--config", "config/features/dev_peptide_peptideclm.yaml"], ["data/dev/peptide_features_peptideclm.csv"]),
                    (["python", "pipeline/features/03_fit_peptide_scaler.py", "--config", "config/features/dev_peptide_peptideclm.yaml"], ["data/dev/peptide_feature_scaler_peptideclm.json"]),
                ],
                "molgraph": [
                    (["python", "pipeline/features/03_fit_peptide_scaler.py", "--config", "config/features/dev_peptide_molgraph.yaml"], ["data/dev/peptide_feature_scaler_molgraph.json"]),
                ],
            },
            "organism": {
                "vocab": [
                    (["python", "pipeline/features/02_build_organism_vocab.py", "--config", "config/features/dev_organism_vocab.yaml"], ["data/dev/organism_vocab_vocab_embedding.json"]),
                ],
            },
        },
    },
}

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{2,48}$")


def _is_axis(v: object, list_typed: bool) -> bool:
    """True when `v` is a grid axis (several values) for this field."""
    if not isinstance(v, list):
        return False
    if list_typed:
        return bool(v) and all(isinstance(x, list) for x in v)
    return True


class _HParams(BaseModel):
    """Base for per-architecture hyperparameter spaces. Subclasses declare each
    field as `T | list[T]`; `axes()` returns {name: [values]} for the grid."""

    model_config = ConfigDict(extra="forbid")

    # fields whose single value is itself a list (so a list of lists is the axis)
    _list_typed: ClassVar[frozenset[str]] = frozenset()

    def axes(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for name in type(self).model_fields:
            v = getattr(self, name)
            if v is None:
                continue
            list_typed = name in self._list_typed
            out[name] = list(v) if _is_axis(v, list_typed) else [v]
        return out


class CommonHParams(BaseModel):
    """Optimiser/loop settings shared by every architecture."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float | list[float] | None = None
    weight_decay: float | list[float] | None = None
    batch_size: int | list[int] | None = None


class BaselineHParams(_HParams):
    organism_embed_dim: int | list[int] | None = None
    hidden_dims: list[int] | list[list[int]] | None = None
    _list_typed: ClassVar[frozenset[str]] = frozenset({"hidden_dims"})


class AttentionFusionHParams(_HParams):
    projection_dim: int | list[int] | None = None
    num_attention_heads: int | list[int] | None = None
    num_attention_layers: int | list[int] | None = None
    hidden_dims: list[int] | list[list[int]] | None = None
    _list_typed: ClassVar[frozenset[str]] = frozenset({"hidden_dims"})


# architecture name -> (short tag used in run names, hyperparameter model)
ARCHITECTURES: dict[str, tuple[str, type[_HParams]]] = {
    "baseline_classifier": ("mlp", BaselineHParams),
    "attention_fusion_classifier": ("attn", AttentionFusionHParams),
}
COMMON_FIELDS = tuple(CommonHParams.model_fields)


class ArchitectureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    # Architecture-specific search space + the shared optimiser fields, one flat dict.
    hparams: dict = {}
    epochs: int | list[int] | None = None

    @field_validator("name")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {v!r}; known: {sorted(ARCHITECTURES)}")
        return v

    @model_validator(mode="after")
    def _validate_hparams(self) -> "ArchitectureRequest":
        _, arch_cls = ARCHITECTURES[self.name]
        common = {k: v for k, v in self.hparams.items() if k in COMMON_FIELDS}
        specific = {k: v for k, v in self.hparams.items() if k not in COMMON_FIELDS}
        # extra="forbid" on both models turns a foreign/typo'd key into an error that
        # names the architecture, instead of silently dropping it.
        try:
            CommonHParams(**common)
            arch_cls(**specific)
        except Exception as e:
            raise ValueError(f"hparams invalid for architecture {self.name!r}: {e}") from e
        return self

    def common(self) -> CommonHParams:
        return CommonHParams(**{k: v for k, v in self.hparams.items() if k in COMMON_FIELDS})

    def specific(self) -> _HParams:
        _, arch_cls = ARCHITECTURES[self.name]
        return arch_cls(**{k: v for k, v in self.hparams.items() if k not in COMMON_FIELDS})


class Representations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peptide: list[str]
    organism: list[str]


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["grid", "random"] = "grid"
    n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _random_needs_n(self) -> "Sampling":
        if (self.mode == "random") != (self.n is not None):
            raise ValueError("`n` is required with mode='random' and not allowed with mode='grid'")
        return self


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str  # W&B group; lowercase slug
    dataset: str = "dev"
    cv: bool = True  # default for (almost) every run; False is rejected until a single-split path exists
    seeds: list[int] = [42]
    representations: Representations
    architectures: list[ArchitectureRequest]
    sampling: Sampling = Sampling()
    snapshot_every_n_epochs: int | None = 2
    # Optional baseline run name for the analysis step to compare against.
    baseline_run: str | None = None

    @field_validator("experiment")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError("experiment must be a lowercase slug (a-z, 0-9, _ or -; 3-49 chars)")
        return v

    @field_validator("cv")
    @classmethod
    def _cv_only(cls, v: bool) -> bool:
        if not v:
            raise ValueError("cv=false is not supported yet (pipeline/train.py single-split runs are not wired into the launcher)")
        return v

    @model_validator(mode="after")
    def _check(self) -> "ExperimentRequest":
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}; known: {sorted(DATASETS)}")
        ds = DATASETS[self.dataset]
        for p in self.representations.peptide:
            if p not in ds["peptide"]:
                raise ValueError(f"peptide representation {p!r} not available for dataset {self.dataset!r}; have {sorted(ds['peptide'])}")
        for o in self.representations.organism:
            if o not in ds["organism"]:
                raise ValueError(f"organism representation {o!r} not available for dataset {self.dataset!r}; have {sorted(ds['organism'])}")
        if not self.architectures:
            raise ValueError("at least one architecture is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        return self
