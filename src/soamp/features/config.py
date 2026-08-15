"""Schema for the features pipeline's config/features/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from soamp.common.tracking import TrackingConfig
from soamp.features.peptide import DESCRIPTOR_NAMES as KNOWN_DESCRIPTOR_NAMES

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

    classification_dataset_filename: str = "mic_classification_dataset.csv"


class DescriptorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = list(KNOWN_DESCRIPTOR_NAMES)

    @field_validator("names", mode="after")
    @classmethod
    def _names_must_be_known(cls, v: list[str]) -> list[str]:
        unknown = [n for n in v if n not in KNOWN_DESCRIPTOR_NAMES]
        if unknown:
            raise ValueError(
                f"unknown descriptor name(s) {unknown}, must be a subset of "
                f"{KNOWN_DESCRIPTOR_NAMES}"
            )
        return v


class OrganismVocabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unknown_index: int = 0


class OutputFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peptide_features_filename: str = "peptide_features.csv"
    organism_vocab_filename: str = "organism_vocab.json"
    peptide_feature_scaler_filename: str = "peptide_feature_scaler.json"


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    input_files: InputFilesConfig = InputFilesConfig()
    descriptors: DescriptorConfig = DescriptorConfig()
    organism_vocab: OrganismVocabConfig = OrganismVocabConfig()
    output_files: OutputFilesConfig = OutputFilesConfig()
    tracking: TrackingConfig = TrackingConfig()
