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
from dataclasses import dataclass
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
    # Seven target strains, MIC recomputed from concentration (ug/ml), NA rows inactive, test + 5 folds
    # from the v2 clusters (docs/dev_dataset_v2.md). All feature files are built locally and shipped in
    # the W&B data artifact (scripts/exp/publish_data.py --dataset dev_v2); `prepare` is only the fallback.
    "dev_v2": {
        "base_config": "cv_dev_v2_base.yaml",
        "peptide": {
            "rdkit": ("peptide_features_rdkit.csv", "peptide_feature_scaler_rdkit.json", "rdkit_descriptors"),
            "peptideclm": ("peptide_features_peptideclm.csv", "peptide_feature_scaler_peptideclm.json", "peptideclm_embedding"),
            "molgraph": ("peptide_features_molgraph.csv", "peptide_feature_scaler_molgraph.json", "molecular_graph"),
        },
        "organism": {
            "vocab": ("organism_vocab_vocab_embedding.json", "vocab_embedding"),
            # ablation: one constant organism index, the model sees only the peptide
            "none": ("organism_vocab_constant.json", "constant"),
        },
        "required_artifacts": ["data/dev_v2/mic_classification_dataset.csv", "data/dev_v2/peptide_split.csv"],
        "prepare": {
            "peptide": {
                "rdkit": [
                    (["python", "pipeline/features/01_build_peptide_features.py", "--config", "config/features/dev_v2_peptide_rdkit.yaml"], ["data/dev_v2/peptide_features_rdkit.csv"]),
                    (["python", "pipeline/features/03_fit_peptide_scaler.py", "--config", "config/features/dev_v2_peptide_rdkit.yaml"], ["data/dev_v2/peptide_feature_scaler_rdkit.json"]),
                ],
                "peptideclm": [
                    (["python", "pipeline/features/01_build_peptide_features.py", "--config", "config/features/dev_v2_peptide_peptideclm.yaml"], ["data/dev_v2/peptide_features_peptideclm.csv"]),
                    (["python", "pipeline/features/03_fit_peptide_scaler.py", "--config", "config/features/dev_v2_peptide_peptideclm.yaml"], ["data/dev_v2/peptide_feature_scaler_peptideclm.json"]),
                ],
                "molgraph": [
                    (["python", "pipeline/features/03_fit_peptide_scaler.py", "--config", "config/features/dev_v2_peptide_molgraph.yaml"], ["data/dev_v2/peptide_feature_scaler_molgraph.json"]),
                ],
            },
            "organism": {
                "vocab": [
                    (["python", "pipeline/features/02_build_organism_vocab.py", "--config", "config/features/dev_v2_organism_vocab.yaml"], ["data/dev_v2/organism_vocab_vocab_embedding.json"]),
                ],
                "none": [
                    (["python", "pipeline/features/02_build_organism_vocab.py", "--config", "config/features/dev_v2_organism_none.yaml"], ["data/dev_v2/organism_vocab_constant.json"]),
                ],
            },
        },
    },
}

DEFAULT_DATASET = "dev_v2"   # a request that names no dataset means the seven-strain data

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


class HistGradientBoostingHParams(_HParams):
    """Tree model: has its OWN learning_rate (not the optimiser's), and `max_iter` rounds
    instead of epochs."""

    max_iter: int | list[int] | None = None
    learning_rate: float | list[float] | None = None
    max_leaf_nodes: int | list[int] | None = None
    min_samples_leaf: int | list[int] | None = None
    l2_regularization: float | list[float] | None = None
    max_features: float | list[float] | None = None


class XGBoostHParams(_HParams):
    """XGBoost tree model: own learning_rate, `n_estimators` boosting rounds instead of epochs."""

    n_estimators: int | list[int] | None = None
    learning_rate: float | list[float] | None = None
    max_depth: int | list[int] | None = None
    min_child_weight: float | list[float] | None = None
    reg_lambda: float | list[float] | None = None
    reg_alpha: float | list[float] | None = None
    gamma: float | list[float] | None = None
    subsample: float | list[float] | None = None
    colsample_bytree: float | list[float] | None = None


class LogisticRegressionHParams(_HParams):
    """Linear baseline (scikit-learn LogisticRegression, L2, lbfgs): ``C`` = inverse regularisation."""

    C: float | list[float] | None = None
    max_iter: int | list[int] | None = None


@dataclass(frozen=True)
class ArchSpec:
    tag: str  # short name used in run names
    hparams_cls: type[_HParams]
    uses_common: bool = True  # shares optim.learning_rate/weight_decay + loop.batch_size/epochs
    vector_only: bool = False  # cannot consume the molecular_graph peptide input
    rounds_param: str | None = None  # trees: the hyperparameter that sets the number of boosting rounds (no epochs)


ARCHITECTURES: dict[str, ArchSpec] = {
    "baseline_classifier": ArchSpec("mlp", BaselineHParams),
    "attention_fusion_classifier": ArchSpec("attn", AttentionFusionHParams),
    "hist_gradient_boosting_classifier": ArchSpec("hgb", HistGradientBoostingHParams, uses_common=False, vector_only=True, rounds_param="max_iter"),
    "xgboost_classifier": ArchSpec("xgb", XGBoostHParams, uses_common=False, vector_only=True, rounds_param="n_estimators"),
    "logistic_regression_classifier": ArchSpec("logreg", LogisticRegressionHParams, uses_common=False, vector_only=True),
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

    def _split(self) -> tuple[dict, dict]:
        """(common optimiser fields, architecture-specific fields)."""
        if not ARCHITECTURES[self.name].uses_common:
            return {}, dict(self.hparams)
        return (
            {k: v for k, v in self.hparams.items() if k in COMMON_FIELDS},
            {k: v for k, v in self.hparams.items() if k not in COMMON_FIELDS},
        )

    @model_validator(mode="after")
    def _validate_hparams(self) -> "ArchitectureRequest":
        spec = ARCHITECTURES[self.name]
        if not spec.uses_common and self.epochs is not None:
            hint = f"; set hparams.{spec.rounds_param} (boosting rounds) instead" if spec.rounds_param else " (it is fitted once)"
            raise ValueError(f"architecture {self.name!r} has no epochs{hint}")
        common, specific = self._split()
        # extra="forbid" on both models turns a foreign/typo'd key into an error that
        # names the architecture, instead of silently dropping it.
        try:
            CommonHParams(**common)
            spec.hparams_cls(**specific)
        except Exception as e:
            raise ValueError(f"hparams invalid for architecture {self.name!r}: {e}") from e
        return self

    def common(self) -> CommonHParams:
        return CommonHParams(**self._split()[0])

    def specific(self) -> _HParams:
        return ARCHITECTURES[self.name].hparams_cls(**self._split()[1])


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
    dataset: str = DEFAULT_DATASET   # "dev_v2" (7 strains, recomputed MIC); "dev" = the older 3-strain data
    cv: bool = True  # default for (almost) every run; False is rejected until a single-split path exists
    seeds: list[int] = [42]
    representations: Representations
    architectures: list[ArchitectureRequest]
    sampling: Sampling = Sampling()
    snapshot_every_n_epochs: int | None = 2
    # vast.ai GPU. Default = the class every earlier successful vast run used (config/vast/dev_*.yaml);
    # a consumer RTX 4090 host answered 403 on api.wandb.ai (2026-11-22), so cheaper cards are opt-in.
    gpu_name: str = "A100_SXM4"
    # Optional baseline run name for the analysis step to compare against.
    baseline_run: str | None = None

    @field_validator("experiment")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError("experiment must be a lowercase slug (a-z, 0-9, _ or -; 3-49 chars)")
        return v

    @field_validator("gpu_name")
    @classmethod
    def _gpu_token(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]{2,32}", v):  # goes into a vast.ai search query string
            raise ValueError("gpu_name must be a vast.ai GPU token like A100_SXM4 or RTX_4090")
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
        if "molgraph" in self.representations.peptide:
            bad = [a.name for a in self.architectures if ARCHITECTURES[a.name].vector_only]
            if bad:
                raise ValueError(f"architecture(s) {bad} need a vector peptide representation, not molgraph")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        return self
