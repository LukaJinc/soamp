"""Schema for the labeling pipeline's config/labeling/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Path(".cache")
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    thresholds_dir: Path = Path("config/thresholds")

    @field_validator("cache_dir", "data_dir", "reports_dir", "thresholds_dir", mode="after")
    @classmethod
    def _resolve_relative_to_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v


class ThresholdTableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "organism_thresholds.csv"


class LabelingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    threshold_table: ThresholdTableConfig = ThresholdTableConfig()
