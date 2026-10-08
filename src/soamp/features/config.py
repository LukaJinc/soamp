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


class MolecularGraphConfig(BaseModel):
    """No options: the graph is a fixed function of the SMILES. The GNN's own
    hyperparameters belong to the model (config/train ... model.graph_encoder)."""

    model_config = ConfigDict(extra="forbid")


PEPTIDE_METHODS = ("rdkit_descriptors", "peptideclm_embedding", "molecular_graph")


class PeptideFeaturizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "rdkit_descriptors"
    rdkit_descriptors: RDKitDescriptorsConfig = RDKitDescriptorsConfig()
    peptideclm_embedding: PeptideCLMConfig = PeptideCLMConfig()
    molecular_graph: MolecularGraphConfig = MolecularGraphConfig()

    @model_validator(mode="after")
    def _method_must_be_known(self) -> "PeptideFeaturizationConfig":
        if self.method not in PEPTIDE_METHODS:
            raise ValueError(f"method must be one of {PEPTIDE_METHODS}, got {self.method!r}")
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


class DnabertSConfig(BaseModel):
    """What the training-time featurizer needs: where the committed embedding
    table lives. The model/extraction settings used to *build* that table are
    in Dnabert16SEmbeddingConfig (organism_16s_embedding block)."""

    model_config = ConfigDict(extra="forbid")

    embeddings_path: str = "data/organism_16s_dnabert_s.json"


ORGANISM_METHODS = ("vocab_embedding", "kmer_composition", "dnabert_s_16s")


class OrganismFeaturizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = "vocab_embedding"
    vocab_embedding: VocabEmbeddingConfig = VocabEmbeddingConfig()
    kmer_composition: KmerCompositionConfig = KmerCompositionConfig()
    dnabert_s_16s: DnabertSConfig = DnabertSConfig()

    @model_validator(mode="after")
    def _method_must_be_known(self) -> "OrganismFeaturizationConfig":
        if self.method not in ORGANISM_METHODS:
            raise ValueError(f"method must be one of {ORGANISM_METHODS}, got {self.method!r}")
        return self

    def active_kwargs(self) -> dict:
        return getattr(self, self.method).model_dump()


class Dnabert16SEmbeddingConfig(BaseModel):
    """Settings for pipeline/features/04_embed_organism_16s_dnabert_s.py -- the
    one-off job that extracts each organism's 16S gene and embeds it. Not read
    at training time."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: str = "zhihan1996/DNABERT-S"
    # Pinned commit of the checkpoint's repo (it ships remote modeling code,
    # executed with trust_remote_code=True -- never track a moving branch).
    revision: str = "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
    max_length: int = 512
    batch_size: int = 8
    device: str = "auto"
    gff_cache_dir: str = ".cache/refseq_gff"
    output_path: str = "data/organism_16s_dnabert_s.json"


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
    organism_16s_embedding: Dnabert16SEmbeddingConfig = Dnabert16SEmbeddingConfig()
    output_files: OutputFilesConfig = OutputFilesConfig()
