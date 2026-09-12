"""Schema for the features pipeline's config/features/base.yaml.

`extra="forbid"` on every model so a typo'd key raises at load time instead
of silently falling back to a default three files away from where it's used.
"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from soamp.features.peptide import DESCRIPTOR_NAMES as KNOWN_DESCRIPTOR_NAMES

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


class RDKitDescriptorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    descriptor_names: list[str] = list(KNOWN_DESCRIPTOR_NAMES)

    @field_validator("descriptor_names", mode="after")
    @classmethod
    def _names_must_be_known(cls, v: list[str]) -> list[str]:
        unknown = [n for n in v if n not in KNOWN_DESCRIPTOR_NAMES]
        if unknown:
            raise ValueError(
                f"unknown descriptor name(s) {unknown}, must be a subset of "
                f"{KNOWN_DESCRIPTOR_NAMES}"
            )
        return v


class PeptideCLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str = "aaronfeller/PeptideCLM-23M-all"
    batch_size: int = 16
    max_length: int = 512
    # Passed to soamp.utils.device.resolve_device. This forward pass is the
    # one GPU-bound step in the pipeline, so "auto" matters here.
    device: str = "auto"


class PeptideFeaturizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "rdkit_descriptors"
    rdkit_descriptors: RDKitDescriptorsConfig = RDKitDescriptorsConfig()
    peptideclm_embedding: PeptideCLMConfig = PeptideCLMConfig()

    @model_validator(mode="after")
    def _method_must_be_known(self) -> "PeptideFeaturizationConfig":
        if self.method not in ("rdkit_descriptors", "peptideclm_embedding"):
            raise ValueError(
                f"method must be 'rdkit_descriptors' or 'peptideclm_embedding', "
                f"got {self.method!r}"
            )
        return self

    def active_kwargs(self) -> dict:
        return getattr(self, self.method).model_dump()


class VocabEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unknown_index: int = 0


class KmerCompositionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_values: list[int] = [1, 2, 3, 4]
    accessions_csv_path: str = "config/organism_genomes/organism_genome_accessions.csv"
    genome_cache_dir: str = ".cache/refseq_genomes"


class OrganismFeaturizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "vocab_embedding"
    vocab_embedding: VocabEmbeddingConfig = VocabEmbeddingConfig()
    kmer_composition: KmerCompositionConfig = KmerCompositionConfig()

    @model_validator(mode="after")
    def _method_must_be_known(self) -> "OrganismFeaturizationConfig":
        if self.method not in ("vocab_embedding", "kmer_composition"):
            raise ValueError(
                f"method must be 'vocab_embedding' or 'kmer_composition', got {self.method!r}"
            )
        return self

    def active_kwargs(self) -> dict:
        return getattr(self, self.method).model_dump()


class OutputFilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peptide_features_filename: str = "peptide_features.csv"
    organism_vocab_filename: str = "organism_vocab.json"
    peptide_feature_scaler_filename: str = "peptide_feature_scaler.json"


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = PathsConfig()
    input_files: InputFilesConfig = InputFilesConfig()
    peptide_featurization: PeptideFeaturizationConfig = PeptideFeaturizationConfig()
    organism_featurization: OrganismFeaturizationConfig = OrganismFeaturizationConfig()
    output_files: OutputFilesConfig = OutputFilesConfig()
