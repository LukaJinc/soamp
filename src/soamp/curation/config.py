"""Schema for the curation pipeline's config/curation/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from soamp.common.tracking import TrackingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Path(".cache")
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    tracking_dir: Path = Path("reports/pipeline_runs")

    @field_validator("cache_dir", "data_dir", "reports_dir", "tracking_dir", mode="after")
    @classmethod
    def _resolve_relative_to_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v


class CrawlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_workers: int = 20
    max_id: int = 24207
    id_margin: int = 500
    request_timeout_s: int = 20


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_threshold: float = 0.60
    test_size: float = 0.2
    random_seed: int = 42
    post_filtering: bool = True


class CurationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    crawl: CrawlConfig = CrawlConfig()
    split: SplitConfig = SplitConfig()
    tracking: TrackingConfig = TrackingConfig()
