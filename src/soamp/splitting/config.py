"""Schema for the splitting pipeline's config/splitting/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    @field_validator("data_dir", "reports_dir", mode="after")
    @classmethod
    def _resolve_relative_to_repo_root(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v


class InputFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification_dataset_filename: str = "mic_classification_dataset.csv"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = "peptide_split.csv"
    sidecar_filename: str = "peptide_split.json"


class FingerprintGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint_threshold: float = 0.8
    radius: int = 2
    n_bits: int = 2048


class QMAPGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_threshold: float = 0.60
    matrix: str = "blosum45"
    gap_open: int = 5
    gap_extension: int = 1
    use_cache: bool = True
    num_threads: int | None = None


class LeidenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_iterations: int = 2
    seed: int = 42


class BucketingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = 0.2
    n_folds: int = 5
    lambda_noncanonical: float = 8.0
    n_iterations: int = 40_000
    seed: int = 42


class SplittingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    input_files: InputFilesConfig = InputFilesConfig()
    output: OutputConfig = OutputConfig()
    fingerprint_graph: FingerprintGraphConfig = FingerprintGraphConfig()
    qmap_graph: QMAPGraphConfig = QMAPGraphConfig()
    leiden: LeidenConfig = LeidenConfig()
    bucketing: BucketingConfig = BucketingConfig()
