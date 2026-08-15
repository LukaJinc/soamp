"""Schema for the dataset-assembly stage's config/data/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from soamp.common.tracking import TrackingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    tracking_dir: Path = Path("reports/pipeline_runs")

    @field_validator("data_dir", "reports_dir", "tracking_dir", mode="after")
    @classmethod
    def _resolve_relative_to_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v


class InputFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regression_dataset_filename: str = "final_mic_regression_dataset.csv"
    activity_labels_filename: str = "mic_activity_labels.csv"
    split_indices_filename: str = "split_indices.json"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "mic_classification_dataset.csv"


class LabelFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    included_labels: list[str] = ["active", "inactive"]


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    input_files: InputFilesConfig = InputFilesConfig()
    output: OutputConfig = OutputConfig()
    label_filter: LabelFilterConfig = LabelFilterConfig()
    tracking: TrackingConfig = TrackingConfig()
