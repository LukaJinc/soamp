"""Schema for the training pipeline's config/train/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    checkpoint_dir: Path = Path("reports/checkpoints")

    @field_validator("data_dir", "reports_dir", "checkpoint_dir", mode="after")
    @classmethod
    def _resolve_relative_to_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v


class InputFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification_dataset_filename: str = "mic_classification_dataset.csv"
    peptide_features_filename: str = "peptide_features.csv"
    organism_vocab_filename: str = "organism_vocab.json"
    peptide_feature_scaler_filename: str = "peptide_feature_scaler.json"
    val_split_filename: str = "val_split.json"


class BaselineClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organism_embed_dim: int = 8
    hidden_dims: list[int] = [32, 16]


class AttentionFusionClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projection_dim: int = 128
    num_attention_heads: int = 4
    num_attention_layers: int = 1
    hidden_dims: list[int] = [64, 32]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str = "baseline_classifier"
    baseline_classifier: BaselineClassifierConfig = BaselineClassifierConfig()
    attention_fusion_classifier: AttentionFusionClassifierConfig = AttentionFusionClassifierConfig()

    @model_validator(mode="after")
    def _architecture_must_be_known(self) -> "ModelConfig":
        if self.architecture not in ("baseline_classifier", "attention_fusion_classifier"):
            raise ValueError(
                f"architecture must be 'baseline_classifier' or "
                f"'attention_fusion_classifier', got {self.architecture!r}"
            )
        return self

    def active_kwargs(self) -> dict:
        return getattr(self, self.architecture).model_dump()


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class ClassBalancingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "auto"
    fixed_pos_weight: float | None = None

    @model_validator(mode="after")
    def _fixed_requires_value(self) -> "ClassBalancingConfig":
        if self.mode not in ("auto", "fixed", "none"):
            raise ValueError(f"mode must be one of auto/fixed/none, got {self.mode!r}")
        if self.mode == "fixed" and self.fixed_pos_weight is None:
            raise ValueError("fixed_pos_weight is required when mode='fixed'")
        return self


class LoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 30
    batch_size: int = 256
    seed: int = 42
    num_dataloader_workers: int = 0
    # Passed to soamp.utils.device.resolve_device -- "auto" picks CUDA when
    # available, so the same config runs on a laptop and a Colab GPU runtime.
    device: str = "auto"


# Which validation metric selects the "best" checkpoint, and whether a lower
# value is better for it. Keys are the `val_*` names pipeline/train.py logs.
BEST_METRIC_LOWER_IS_BETTER = {
    "val_auroc": False,
    "val_accuracy": False,
    "val_f1": False,
    "val_precision": False,
    "val_recall": False,
    "val_loss": True,
}


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_every_n_epochs: int = 5
    best_metric: str = "val_auroc"

    @field_validator("best_metric", mode="after")
    @classmethod
    def _best_metric_must_be_known(cls, v: str) -> str:
        if v not in BEST_METRIC_LOWER_IS_BETTER:
            raise ValueError(
                f"best_metric must be one of "
                f"{sorted(BEST_METRIC_LOWER_IS_BETTER)}, got {v!r}"
            )
        return v

    @property
    def lower_is_better(self) -> bool:
        return BEST_METRIC_LOWER_IS_BETTER[self.best_metric]


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Sequence-keyed fold assignments (soamp.engine.cross_validation reads
    # this by "sequence", not "node_id"/"peptide_id" -- see that module's
    # docstring for why the other two are the wrong join key).
    folds_csv_filename: str = "train_folds_leiden.csv"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exp_id: str = "baseline_mlp_v1"
    paths: PathsConfig = PathsConfig()
    input_files: InputFilesConfig = InputFilesConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    class_balancing: ClassBalancingConfig = ClassBalancingConfig()
    loop: LoopConfig = LoopConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    # Only read by pipeline/train_cv.py -- pipeline/train.py (the single
    # held-out-split run) never touches this.
    cv: CVConfig = CVConfig()
    # Forwarded to wandb.init via build_tracker's **wandb_init_kwargs, so a
    # set of related runs (e.g. one featurization grid) is grouped/filterable
    # in the UI without any tracking-layer change.
    wandb_group: str | None = None
    wandb_tags: list[str] = []
