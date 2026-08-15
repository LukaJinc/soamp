"""Schema for the training pipeline's config/train/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from soamp.common.tracking import TrackingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    checkpoint_dir: Path = Path("reports/checkpoints")
    tracking_dir: Path = Path("reports/train_runs")

    @field_validator("data_dir", "reports_dir", "checkpoint_dir", "tracking_dir", mode="after")
    @classmethod
    def _resolve_relative_to_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v


class InputFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification_dataset_filename: str = "mic_classification_dataset.csv"
    peptide_features_filename: str = "peptide_features.csv"
    organism_vocab_filename: str = "organism_vocab.json"
    peptide_feature_scaler_filename: str = "peptide_feature_scaler.json"


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = 0.1


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organism_embed_dim: int = 8
    hidden_dims: list[int] = [32, 16]


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


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_every_n_epochs: int = 5
    best_metric: str = "val_auroc"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    input_files: InputFilesConfig = InputFilesConfig()
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    class_balancing: ClassBalancingConfig = ClassBalancingConfig()
    loop: LoopConfig = LoopConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    tracking: TrackingConfig = TrackingConfig()
